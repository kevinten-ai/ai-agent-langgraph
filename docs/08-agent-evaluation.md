# 08 Agent 评估体系

> 本章介绍如何为 LangGraph Agent 构建系统化的评估体系，涵盖 RAGAS 指标、节点级断言、自定义 Eval Pipeline 以及 CI 集成。

---

## 目录

1. [为什么 Agent 需要独立评估](#1-为什么-agent-需要独立评估)
2. [RAGAS 基础指标](#2-ragas-基础指标)
3. [LangGraph 节点级断言评估](#3-langgraph-节点级断言评估)
4. [自定义 Eval Pipeline](#4-自定义-eval-pipeline)
5. [CI 中集成 eval](#5-ci-中集成-eval)

---

## 1. 为什么 Agent 需要独立评估

### 1.1 传统单元测试的局限性

传统软件的单元测试基于"给定输入 -> 确定输出"的假设，但 LLM-based Agent 具有以下特点：

- **非确定性**：相同输入可能产生不同输出
- **多步推理**：中间步骤的错误会级联传播
- **工具调用**：外部依赖使结果难以预测
- **长链条**：LangGraph 的节点跳转形成了复杂的执行轨迹（trajectory）

### 1.2 Agent 评估的三个层面

| 层面 | 评估对象 | 关键问题 |
|------|----------|----------|
| **节点级** | 单个 LangGraph 节点 | 任务分配是否正确？工具选择是否合适？ |
| **轨迹级** | 完整执行路径 | 是否按预期跳转？重试次数是否合理？ |
| **输出级** | 最终答案 | 答案是否忠实于上下文？是否回答了用户问题？ |

### 1.3 评估驱动开发（Eval-Driven Development）

建议在每个 Agent 节点变更后，运行以下检查：

1. 节点断言：状态字段是否符合预期
2. 端到端：用固定 dataset 跑完整工作流
3. 阈值告警：核心指标下降时阻塞合并

---

## 2. RAGAS 基础指标

[RAGAS](https://docs.ragas.io/) 是一套无需人工标注即可评估 RAG 系统的指标库。对于典型的"检索 + 生成" Agent，我们重点关注以下三个指标。

### 2.1 Faithfulness（忠实度）

**定义**：生成的答案是否可以从检索到的上下文中推断出来。避免模型" hallucination（幻觉）"。

**计算方式**：
- 从答案中提取陈述句
- 判断每个陈述是否被上下文支持
- 得分 = 被支持的陈述数 / 总陈述数

### 2.2 Context Relevancy（上下文相关性）

**定义**：检索到的上下文与用户问题的相关程度。用于评估检索模块质量。

**计算方式**：
- 将上下文拆分为句子
- 判断每个句子对回答问题是否必要
- 得分 = 必要句子数 / 总句子数

### 2.3 Answer Relevancy（答案相关性）

**定义**：生成的答案是否直接回答了用户提出的问题，而不是答非所问。

**计算方式**：
- 根据答案反推可能的用户问题
- 比较反推问题与原始问题的语义相似度
- 得分 = 平均相似度

### 2.4 RAGAS 快速使用示例

```bash
# 安装 ragas（建议单独环境）
pip install ragas
```

```python
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas import evaluate

# 构建评估数据集
eval_data = Dataset.from_dict({
    "question": ["LangGraph 是什么？"],
    "answer": ["LangGraph 是用于构建多 Agent 系统的编排框架。"],
    "contexts": [["LangGraph 提供图结构来编排 LLM 应用的工作流。"]],
})

# 运行评估
result = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_relevancy]
)
print(result)
```

> 注意：RAGAS 默认需要调用 LLM（如 OpenAI）来做判断，因此会产生额外的 API 费用。

完整可运行示例请参考：`examples/platform/ragas_evaluation.py`

---

## 3. LangGraph 节点级断言评估

### 3.1 为什么需要节点级断言

RAGAS 只能评估最终输出，但 Agent 的**中间过程**同样关键。LangGraph 的每个节点都可以被单独断言，这能帮助我们快速定位问题所在节点。

### 3.2 节点断言设计思路

以下是一个四节点 LangGraph 工作流的断言示例：

```python
import pytest
from src.workflow import MultiAgentWorkflow
from src.models import TaskType, ExecutionStatus


@pytest.fixture
async def workflow():
    return MultiAgentWorkflow(enable_mcp_integration=False)


@pytest.mark.asyncio
async def test_task_assigner_node_produces_valid_task(workflow):
    """断言：task_assigner 节点必须产出有效的 task_type 和 task_description"""
    state = await workflow._task_assigner_node(
        create_initial_state("设计一个用户管理系统")
    )

    assert state.task_type is not None
    assert state.task_description is not None
    assert len(state.task_description) > 10


@pytest.mark.asyncio
async def test_tool_selector_node_returns_expected_tools(workflow):
    """断言：tool_selector 节点为数据类任务选择 analytics 工具"""
    from src.models import create_initial_state, Task

    state = create_initial_state("分析本月销售数据")
    state.task = Task(
        id="t1",
        user_input="分析本月销售数据",
        task_type=TaskType.DATA_ANALYSIS,
        task_description="分析销售数据",
    )

    state = await workflow._tool_selector_node(state)
    tool_names = [t.tool_name for t in state.selected_tools]

    # 数据类任务至少应选中一个分析相关工具
    assert any("analytics" in name.lower() or "data" in name.lower() for name in tool_names)


@pytest.mark.asyncio
async def test_executor_node_returns_success_status(workflow):
    """断言：executor 节点正常执行后状态为 SUCCESS"""
    from src.models import create_initial_state

    state = create_initial_state("简单的问候")
    state = await workflow._executor_node(state)

    assert state.execution_status == ExecutionStatus.SUCCESS
    assert state.final_answer is not None


@pytest.mark.asyncio
async def test_reviewer_node_score_in_valid_range(workflow):
    """断言：reviewer 节点评分必须在 0-10 之间"""
    from src.models import create_initial_state, ExecutionResult, ExecutionStatus

    state = create_initial_state("测试")
    state.execution_result = ExecutionResult(
        content="测试答案", status=ExecutionStatus.SUCCESS
    )

    state = await workflow._reviewer_node(state)

    assert state.review_score is not None
    assert 0 <= state.review_score <= 10
```

### 3.3 状态转换断言

除了单节点断言，还应验证图结构中的状态转换是否合法：

```python
from src.models import validate_state_transition


def test_state_transition_executor_to_reviewer():
    from_state = create_initial_state("问个好")
    to_state = create_initial_state("问个好")
    to_state.execution_result = ExecutionResult(
        content="你好！", status=ExecutionStatus.SUCCESS
    )
    to_state.execution_status = ExecutionStatus.SUCCESS

    assert validate_state_transition(from_state, to_state, "executor->reviewer")
```

---

## 4. 自定义 Eval Pipeline

### 4.1 什么时候需要自定义 Pipeline

当 RAGAS 无法覆盖你的业务场景时（例如：需要评估工具调用正确性、输出格式 JSON 合规性、回答长度等），可以构建自定义 Eval Pipeline。

### 4.2 Pipeline 架构

```
         +------------------+
         |   Agent 执行完毕  |
         +--------+---------+
                  |
         +--------v---------+
         |  extract_trace() |  <-- 从 state 中提取轨迹
         +--------+---------+
                  |
         +--------v---------+
         |   run_evaluators()|  <-- 并行运行多个评估器
         +--------+---------+
                  |
         +--------v---------+
         | generate_report() |  <-- 汇总为评分报告 dict
         +------------------+
```

### 4.3 自定义评估器示例代码

```python
import json
import re
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class EvalResult:
    """单个评估器结果"""
    name: str
    score: float          # 0.0 ~ 1.0
    passed: bool
    reason: str


class AgentEvalPipeline:
    """自定义 Agent 评估 Pipeline"""

    def __init__(self):
        self.evaluators: List[Callable[[Dict[str, Any]], EvalResult]] = []

    def add(self, evaluator: Callable[[Dict[str, Any]], EvalResult]):
        self.evaluators.append(evaluator)
        return self

    def run(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        results = [ev(trace) for ev in self.evaluators]
        total_score = sum(r.score for r in results) / len(results) if results else 0.0
        all_passed = all(r.passed for r in results)

        return {
            "total_score": round(total_score, 4),
            "passed": all_passed,
            "details": [
                {
                    "metric": r.name,
                    "score": r.score,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in results
            ],
        }


# ------------------ 内置评估器 ------------------

def tool_call_correctness_eval(expected_tools: List[str]) -> Callable:
    """评估工具调用是否与预期一致（允许额外调用）"""
    def _eval(trace: Dict[str, Any]) -> EvalResult:
        called = set(trace.get("called_tools", []))
        expected = set(expected_tools)
        hit = len(expected & called)
        score = hit / len(expected) if expected else 1.0
        return EvalResult(
            name="tool_call_correctness",
            score=score,
            passed=score >= 0.8,
            reason=f"预期调用 {expected_tools}，实际调用 {list(called)}",
        )
    return _eval


def json_format_eval(trace: Dict[str, Any]) -> EvalResult:
    """评估最终答案是否为合法 JSON"""
    answer = trace.get("final_answer", "")
    try:
        json.loads(answer)
        return EvalResult(
            name="json_format",
            score=1.0,
            passed=True,
            reason="最终答案为合法 JSON",
        )
    except json.JSONDecodeError:
        return EvalResult(
            name="json_format",
            score=0.0,
            passed=False,
            reason="最终答案不是合法 JSON",
        )


def citation_eval(trace: Dict[str, Any]) -> EvalResult:
    """评估回答中是否包含引用来源标记，如 [1]、[source] 等"""
    answer = trace.get("final_answer", "")
    has_citation = bool(re.search(r"\[\d+\]|\[source\]|\(来源", answer, re.IGNORECASE))
    return EvalResult(
        name="citation_presence",
        score=1.0 if has_citation else 0.0,
        passed=has_citation,
        reason="检测到引用标记" if has_citation else "未检测到引用标记",
    )


def length_eval(min_len: int = 20, max_len: int = 2000) -> Callable:
    """评估回答长度是否在合理区间"""
    def _eval(trace: Dict[str, Any]) -> EvalResult:
        answer = trace.get("final_answer", "")
        length = len(answer)
        if min_len <= length <= max_len:
            score = 1.0
        elif length < min_len:
            score = length / min_len
        else:
            score = max(0.0, 1.0 - (length - max_len) / max_len)
        return EvalResult(
            name="answer_length",
            score=round(score, 4),
            passed=min_len <= length <= max_len,
            reason=f"回答长度 {length}，要求范围 [{min_len}, {max_len}]",
        )
    return _eval


def no_error_eval(trace: Dict[str, Any]) -> EvalResult:
    """评估执行过程中是否没有错误"""
    errors = trace.get("error_messages", [])
    return EvalResult(
        name="no_error",
        score=1.0 if not errors else 0.0,
        passed=not errors,
        reason="无错误" if not errors else f"发现 {len(errors)} 个错误: {errors}",
    )


# ------------------ 使用示例 ------------------

if __name__ == "__main__":
    # 模拟一次 Agent 执行后的轨迹
    mock_trace = {
        "called_tools": ["search_web", "calculator"],
        "final_answer": json.dumps({"result": 42, "explanation": "答案来自计算和检索。"}),
        "error_messages": [],
    }

    pipeline = AgentEvalPipeline()
    pipeline.add(tool_call_correctness_eval(expected_tools=["search_web"]))
    pipeline.add(json_format_eval)
    pipeline.add(citation_eval)
    pipeline.add(length_eval(min_len=10, max_len=500))
    pipeline.add(no_error_eval)

    report = pipeline.run(mock_trace)
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

完整可运行示例请参考：`examples/platform/custom_eval_pipeline.py`

---

## 5. CI 中集成 eval

### 5.1 设计目标

- **自动化**：每次代码提交自动运行评估
- **可重复**：使用固定的评估数据集（dataset）
- **可量化**：设定指标阈值（threshold），低于阈值时 CI 失败

### 5.2 目录结构建议

```
ai-agent-langgraph/
├── tests/
│   ├── test_nodes.py              # 节点级断言
│   └── test_e2e.py                # 端到端测试
├── eval/
│   ├── __init__.py
│   ├── dataset.json               # 评估数据集
│   ├── pipeline.py                # Eval Pipeline 封装
│   └── thresholds.yaml            # 指标阈值配置
├── .github/
│   └── workflows/
│       └── eval.yml               # CI 工作流
└── pytest.ini
```

### 5.3 评估数据集（dataset.json）示例

```json
[
  {
    "id": "eval_001",
    "input": "计算 15 * 23 并告诉我结果",
    "expected_tools": ["calculator"],
    "expected_answer_contains": ["345"],
    "metrics": {
      "tool_call_correctness": 1.0,
      "answer_relevancy": 0.8
    }
  },
  {
    "id": "eval_002",
    "input": "搜索一下 Python 的最新版本",
    "expected_tools": ["web_search"],
    "expected_answer_contains": ["Python", "3."],
    "metrics": {
      "tool_call_correctness": 1.0,
      "answer_relevancy": 0.8
    }
  }
]
```

### 5.4 pytest + dataset + threshold 测试示例

```python
# tests/test_e2e.py
import json
import pytest
from src.workflow import MultiAgentWorkflow
from examples.platform.custom_eval_pipeline import AgentEvalPipeline, tool_call_correctness_eval, answer_relevancy_eval

# 加载评估数据集
with open("eval/dataset.json", "r", encoding="utf-8") as f:
    EVAL_DATASET = json.load(f)

# 加载阈值配置
THRESHOLDS = {
    "tool_call_correctness": 0.8,
    "answer_relevancy": 0.7,
    "total_score": 0.75,
}


@pytest.fixture(scope="module")
def workflow():
    return MultiAgentWorkflow(enable_mcp_integration=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", EVAL_DATASET)
async def test_eval_case(workflow, case):
    """端到端评估测试：每个 dataset case 都必须满足 threshold"""
    # 1. 运行 Agent
    result = await workflow.run_workflow(case["input"])
    assert result["success"], f"工作流执行失败: {result.get('error')}"

    # 2. 提取轨迹
    summary = result.get("execution_summary", {})
    trace = {
        "input": case["input"],
        "called_tools": [t.tool_name for t in result.get("selected_tools", [])],
        "final_answer": result["final_answer"],
        "error_messages": result.get("error_messages", []),
    }

    # 3. 运行 Eval Pipeline
    pipeline = AgentEvalPipeline()
    pipeline.add(tool_call_correctness_eval(case.get("expected_tools", [])))
    pipeline.add(answer_relevancy_eval)
    report = pipeline.run(trace)

    # 4. 逐指标检查 threshold
    for metric, threshold in THRESHOLDS.items():
        if metric == "total_score":
            assert report["total_score"] >= threshold, (
                f"Case {case['id']} total_score {report['total_score']} < {threshold}"
            )
        else:
            detail = next((d for d in report["details"] if d["metric"] == metric), None)
            if detail:
                assert detail["score"] >= threshold, (
                    f"Case {case['id']} {metric} {detail['score']} < {threshold}"
                )
```

### 5.5 GitHub Actions CI 配置示例

```yaml
# .github/workflows/eval.yml
name: Agent Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ragas pytest pytest-asyncio

      - name: Run node-level tests
        run: pytest tests/test_nodes.py -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Run E2E eval
        run: pytest tests/test_e2e.py -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 5.6 阈值管理建议

| 指标 | 开发阶段阈值 | 生产阶段阈值 | 说明 |
|------|-------------|-------------|------|
| `faithfulness` | 0.70 | 0.85 | 防止幻觉 |
| `answer_relevancy` | 0.70 | 0.80 | 避免答非所问 |
| `tool_call_correctness` | 0.80 | 0.90 | 工具调用准确率 |
| `json_format` | 1.00 | 1.00 | 格式必须完全合规 |
| `total_score` | 0.75 | 0.85 | 综合得分 |

---

## 总结

本章介绍了 Agent 评估的完整体系：

1. **独立评估的必要性**：LLM 的非确定性决定了传统单元测试不够用
2. **RAGAS 指标**：Faithfulness、Context Relevancy、Answer Relevancy 是 RAG Agent 的基础指标
3. **节点级断言**：针对 LangGraph 每个节点做状态断言，快速定位问题
4. **自定义 Pipeline**：通过轨迹评分 + 输出格式校验覆盖业务特有评估需求
5. **CI 集成**：用 `pytest + dataset + threshold` 的模式实现自动化、可量化评估

建议在实际项目中：
- 先写节点级测试，保证图结构的稳定性
- 再构建 dataset，覆盖常见用户问题
- 最后将 Eval Pipeline 接入 CI，守护代码质量

---

*参考资料*：
- [RAGAS 官方文档](https://docs.ragas.io/)
- [LangGraph 测试最佳实践](https://langchain-ai.github.io/langgraph/concepts/testing/)
