#!/usr/bin/env python3
"""
自定义 Eval Pipeline 示例

不依赖复杂第三方库，对一个 LangGraph Agent 的执行轨迹进行自定义评分。
评估维度包括：
    - 工具调用正确性
    - 输出格式是否为 JSON
    - 是否有引用来源
    - 回答长度是否合理
    - 执行过程中是否有错误

代码可直接运行，无需额外安装非标准库（仅用 Python 内置模块）。
"""

import json
import re
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class EvalResult:
    """单个评估器的结果"""
    name: str
    score: float          # 0.0 ~ 1.0
    passed: bool
    reason: str


class AgentEvalPipeline:
    """
    自定义 Agent 评估 Pipeline。

    使用方式：
        pipeline = AgentEvalPipeline()
        pipeline.add(tool_call_correctness_eval(expected_tools=["calculator"]))
        pipeline.add(json_format_eval)
        report = pipeline.run(trace)
    """

    def __init__(self):
        self.evaluators: List[Callable[[Dict[str, Any]], EvalResult]] = []

    def add(self, evaluator: Callable[[Dict[str, Any]], EvalResult]):
        """向 Pipeline 中添加一个评估器"""
        self.evaluators.append(evaluator)
        return self

    def run(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行所有评估器并生成评分报告。

        Args:
            trace: Agent 执行后的轨迹字典，需包含 pipeline 中各评估器所需的字段。

        Returns:
            包含 total_score、passed、details 的报告字典
        """
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


# ============================================================
# 预置评估器
# ============================================================

def tool_call_correctness_eval(expected_tools: List[str]) -> Callable:
    """
    工具调用正确性评估器。

    判断 Agent 是否调用了预期的工具（允许额外调用，但不允许缺失）。
    """
    def _eval(trace: Dict[str, Any]) -> EvalResult:
        called = set(trace.get("called_tools", []))
        expected = set(expected_tools)
        hit = len(expected & called)
        score = hit / len(expected) if expected else 1.0
        return EvalResult(
            name="tool_call_correctness",
            score=round(score, 4),
            passed=score >= 0.8,
            reason=f"预期调用 {expected_tools}，实际调用 {list(called)}",
        )
    return _eval


def output_format_eval(expected_format: str = "json") -> Callable:
    """
    输出格式评估器。

    目前支持判断最终答案是否为合法 JSON。
    后续可扩展为验证 XML、YAML 等格式。
    """
    def _eval(trace: Dict[str, Any]) -> EvalResult:
        answer = trace.get("final_answer", "")
        if expected_format == "json":
            try:
                json.loads(answer)
                return EvalResult(
                    name="output_format",
                    score=1.0,
                    passed=True,
                    reason="最终答案为合法 JSON",
                )
            except json.JSONDecodeError:
                return EvalResult(
                    name="output_format",
                    score=0.0,
                    passed=False,
                    reason="最终答案不是合法 JSON",
                )
        return EvalResult(
            name="output_format",
            score=1.0,
            passed=True,
            reason=f"未对格式 {expected_format} 做校验",
        )
    return _eval


def citation_eval(trace: Dict[str, Any]) -> EvalResult:
    """
    引用来源评估器。

    检测最终答案中是否包含常见的引用标记，如 [1]、[source] 等。
    """
    answer = trace.get("final_answer", "")
    has_citation = bool(re.search(r"\[\d+\]|\[source\]|\(来源：?|参考：", answer, re.IGNORECASE))
    return EvalResult(
        name="citation_presence",
        score=1.0 if has_citation else 0.0,
        passed=has_citation,
        reason="检测到引用标记" if has_citation else "未检测到引用标记",
    )


def answer_length_eval(min_len: int = 20, max_len: int = 2000) -> Callable:
    """
    回答长度评估器。

    判断最终答案长度是否在 [min_len, max_len] 范围内。
    超出范围时按偏离程度计算部分得分。
    """
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
    """错误检测评估器：判断执行过程中是否没有任何错误信息。"""
    errors = trace.get("error_messages", [])
    return EvalResult(
        name="no_error",
        score=1.0 if not errors else 0.0,
        passed=not errors,
        reason="无错误" if not errors else f"发现 {len(errors)} 个错误: {errors}",
    )


def retry_count_eval(max_retries: int = 1) -> Callable:
    """
    重试次数评估器。

    判断 Agent 的工作流重试次数是否不超过阈值，
    重试过多通常意味着节点逻辑不稳定或 Prompt 设计不佳。
    """
    def _eval(trace: Dict[str, Any]) -> EvalResult:
        retries = trace.get("retry_count", 0)
        passed = retries <= max_retries
        score = 1.0 if passed else max(0.0, 1.0 - (retries - max_retries) * 0.3)
        return EvalResult(
            name="retry_count",
            score=round(score, 4),
            passed=passed,
            reason=f"重试次数 {retries}，阈值 {max_retries}",
        )
    return _eval


# ============================================================
# 辅助函数：从 LangGraph Agent 结果中提取 trace
# ============================================================

def extract_trace_from_agent_result(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 src.workflow.MultiAgentWorkflow.run_workflow 的返回结果中提取评估所需的 trace。

    Args:
        agent_result: run_workflow 返回的字典

    Returns:
        供 AgentEvalPipeline.run 使用的 trace 字典
    """
    summary = agent_result.get("execution_summary", {})
    return {
        "input": agent_result.get("final_answer", ""),
        "called_tools": summary.get("selected_tools", []),
        "final_answer": agent_result.get("final_answer", ""),
        "error_messages": agent_result.get("error_messages", []),
        "retry_count": summary.get("retry_count", 0),
    }


# ============================================================
# 主函数：演示如何对模拟轨迹进行评估
# ============================================================

def main():
    print("=" * 60)
    print("自定义 Agent Eval Pipeline 演示")
    print("=" * 60)

    # 模拟一次 Agent 执行后的轨迹（实际项目中可从 run_workflow 结果中提取）
    mock_trace = {
        "called_tools": ["search_web", "calculator"],
        "final_answer": json.dumps({
            "result": 42,
            "explanation": "答案来自计算和检索。[source: web]",
        }, ensure_ascii=False),
        "error_messages": [],
        "retry_count": 0,
    }

    # 构建 Pipeline 并注册评估器
    pipeline = AgentEvalPipeline()
    pipeline.add(tool_call_correctness_eval(expected_tools=["search_web"]))
    pipeline.add(output_format_eval(expected_format="json"))
    pipeline.add(citation_eval)
    pipeline.add(answer_length_eval(min_len=10, max_len=500))
    pipeline.add(no_error_eval)
    pipeline.add(retry_count_eval(max_retries=1))

    # 运行评估
    report = pipeline.run(mock_trace)

    # 输出报告
    print("\n📊 评分报告:")
    print(f"   总得分 (total_score): {report['total_score']}")
    print(f"   是否通过 (passed): {report['passed']}")
    print("\n   各指标详情:")
    for detail in report["details"]:
        status = "通过" if detail["passed"] else "未通过"
        print(
            f"   - {detail['metric']:20s} 得分: {detail['score']:.2f}  [{status}]  {detail['reason']}"
        )

    # 第二个演示：失败场景（JSON 不合法 + 缺少工具 + 有错误）
    print("\n" + "=" * 60)
    print("失败场景演示")
    print("=" * 60)

    bad_trace = {
        "called_tools": ["calculator"],
        "final_answer": "这不是 JSON 格式",
        "error_messages": ["工具 search_web 调用超时"],
        "retry_count": 2,
    }

    bad_report = pipeline.run(bad_trace)
    print(f"\n📊 评分报告:")
    print(f"   总得分 (total_score): {bad_report['total_score']}")
    print(f"   是否通过 (passed): {bad_report['passed']}")
    print("\n   各指标详情:")
    for detail in bad_report["details"]:
        status = "通过" if detail["passed"] else "未通过"
        print(
            f"   - {detail['metric']:20s} 得分: {detail['score']:.2f}  [{status}]  {detail['reason']}"
        )


if __name__ == "__main__":
    main()
