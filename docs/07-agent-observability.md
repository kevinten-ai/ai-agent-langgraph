# 07 - Agent 可观测性（Observability）

> 本章节介绍如何为 LangGraph 构建的多 Agent 工作流添加可观测性，涵盖 LangSmith、OpenTelemetry 以及自定义指标上报等核心实践。

---

## 目录

1. [为什么 Agent 需要可观测性](#1-为什么-agent-需要可观测性)
2. [LangSmith Trace 基础配置](#2-langsmith-trace-基础配置)
3. [`astream_events` 结构化日志采集](#3-astream_events-结构化日志采集)
4. [OpenTelemetry Trace 集成概念](#4-opentelemetry-trace-集成概念)
5. [自定义 Node 级 Metric 上报思路](#5-自定义-node-级-metric-上报思路)

---

## 1. 为什么 Agent 需要可观测性

相比传统 REST API，基于 LLM 的 Agent 系统具有**非确定性（Nondeterministic）**、**多步推理**、**工具调用链长**等特点。
生产环境中如果没有可观测性，排查问题如同“黑盒”：

| 痛点 | 可观测性提供的价值 |
|------|------------------|
| LLM 输出不可预期 | 通过 Trace 查看每一步的 Prompt 和 Response，快速定位异常 |
| 工具调用失败 | 可视化节点执行顺序，查看哪个 Node/MCP 工具超时或报错 |
| 延迟抖动 | 分析每个节点的耗时（Latency），找到性能瓶颈 |
| Token 成本不可控 | 统计各阶段 Token 消耗，辅助成本优化 |
| 重试/循环导致死锁 | 通过状态流转图观察循环次数，及时调整条件边逻辑 |

简言之，可观测性 = **Trace（调用链）+ Metrics（指标）+ Logs（日志）**。LangGraph 作为图编排框架，天然支持事件流，让我们可以在这三个维度上做文章。

---

## 2. LangSmith Trace 基础配置

LangSmith 是 LangChain 官方提供的可观测性平台，与 LangGraph 无缝集成。开启方式非常简单：

### 2.1 环境变量配置

```bash
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY="ls-xxxxxxxxxxxxxxxxxxxx"
export LANGCHAIN_PROJECT="ai-agent-langgraph-prod"  # 可选，用于项目分组
```

| 环境变量 | 说明 |
|---------|------|
| `LANGCHAIN_TRACING_V2` | 开启 LangSmith Tracing（布尔值） |
| `LANGSMITH_API_KEY` | 你的 LangSmith API Key |
| `LANGCHAIN_PROJECT` | 指定项目名，方便在 LangSmith UI 中分类查看 |

### 2.2 Python 代码中的额外控制

如果只想在部分请求中开启 Trace，可以在 `config` 中显式传入 `callbacks` 或元数据：

```python
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer(project_name="ai-agent-langgraph-dev")

config = {
    "callbacks": [tracer],
    "metadata": {"user_id": "user-123", "session_id": "sess-456"},
    "tags": ["multi-agent", "v2"],
}

# 调用图时传入 config
result = await app.ainvoke(input_state, config=config)
```

### 2.3 与 `orchestrator.py` 集成

`src/workflow/orchestrator.py` 中通过 `get_graph()` 暴露编译后的图。由于 LangSmith 会自动对 `langchain` 和 `langgraph` 中的 Runnable 进行追踪，因此只要设置了环境变量，调用 `app.ainvoke()` 或 `app.astream_events()` 时就会自动产生 Trace，无需修改业务代码。

示例：

```python
import asyncio
import os
from src.workflow.orchestrator import get_graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

async def main():
    app = get_graph()
    # 这里会自动产生 LangSmith trace
    result = await app.ainvoke({"user_input": "帮我查一下今天的天气"})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

> **注意**：如果没有设置有效的 `LANGSMITH_API_KEY`，程序依然可以正常运行，只是不会向 LangSmith 远端上报 Trace。

---

## 3. `astream_events` 结构化日志采集

LangGraph 提供了 `astream_events(version="v2")` API，它能在图的运行过程中以**细粒度事件流**的形式输出内部状态。这些事件本身就是极佳的“结构化日志源”。

### 3.1 事件类型速查

| 事件名 | 触发时机 |
|--------|---------|
| `on_chain_start` / `on_chain_end` | 每个 Node（包括子图）开始/结束 |
| `on_chat_model_start` / `on_chat_model_end` | LLM 调用开始/结束 |
| `on_chat_model_stream` | LLM 流式输出单个 token |
| `on_tool_start` / `on_tool_end` | Tool（包括 MCP 工具）调用开始/结束 |
| `on_prompt_start` / `on_prompt_end` | Prompt 模板渲染开始/结束 |

### 3.2 采集示例

以下代码演示如何从 `astream_events` 中提取关键信息，构建自定义日志：

```python
async def collect_events(app, input_state):
    logs = []

    async for event in app.astream_events(input_state, version="v2"):
        event_type = event["event"]
        name = event.get("name", "unknown")
        run_id = event.get("run_id")
        metadata = event.get("metadata", {})
        data = event.get("data", {})

        # 节点级别生命周期事件
        if event_type in ("on_chain_start", "on_chain_end") and name != "LangGraph":
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "event": event_type,
                "node": name,
                "run_id": str(run_id),
                "metadata": metadata,
            })

        # LLM token 流事件（可用于后续 token 计数）
        elif event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            token = chunk.content if chunk else ""
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "DEBUG",
                "event": "llm_token",
                "token": token,
                "run_id": str(run_id),
            })

        # 工具调用事件
        elif event_type == "on_tool_end":
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "event": "tool_end",
                "tool": name,
                "output": str(data.get("output")),
                "run_id": str(run_id),
            })

    return logs
```

### 3.3 与 orchestrator 集成的安全提示

由于 `orchestrator.py` 中的 `AgentState` 是一个 `TypedDict`（且内部包含 Pydantic 模型），某些字段在默认 JSON 序列化时可能会报错。建议在记录日志前对复杂对象做降级处理：

```python
def safe_serialize(obj):
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)
    except Exception:
        return "<unserializable>"
```

---

## 4. OpenTelemetry Trace 集成概念

LangSmith 提供开箱即用的体验，但在某些企业场景中，团队可能已经建立了基于 **OpenTelemetry（OTel）** 的可观测体系（如 Jaeger、Datadog、AWS X-Ray）。此时可以将 LangGraph 的执行过程以 OTel Span 的形式接入现有链路。

### 4.1 核心概念

- **TracerProvider**：全局的 Trace 生产者。
- **Tracer**：用于创建 Span 的实例，通常按库名区分。
- **Span**：代表一次操作（如一个 Node 的执行、一次 LLM 调用）。
- **Context / Propagator**：用于跨进程/跨服务传递 Trace ID。

### 4.2 手动为 LangGraph Node 添加 Span

目前 LangGraph 尚未原生暴露 OTel 自动埋点接口，但我们可以通过装饰器或显式代码在 Node 函数内部添加 Span。

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.status import Status, StatusCode

# 1. 初始化 Provider 和 Exporter
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 2. 获取 Tracer
tracer = trace.get_tracer("ai-agent-langgraph")

# 3. 在 Node 中手动创建 Span
async def task_assigner_node(state):
    with tracer.start_as_current_span("node.task_assigner") as span:
        span.set_attribute("node.type", "task_assigner")
        span.set_attribute("input.preview", state.get("user_input", "")[:100])

        try:
            # 原有业务逻辑...
            result = await some_async_work(state)
            span.set_attribute("output.status", "success")
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

### 4.3 将事件流映射到 Span

更高级的做法是：在 `astream_events` 的循环中，根据 `on_chain_start` / `on_chain_end` 自动开启和关闭 Span，实现**无侵入式**的 OTel 集成：

```python
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("ai-agent-langgraph")
active_spans = {}

async for event in app.astream_events(input_state, version="v2"):
    event_type = event["event"]
    name = event.get("name", "unknown")
    run_id = event.get("run_id")

    if event_type == "on_chain_start" and name != "LangGraph":
        span = tracer.start_span(
            name=f"node.{name}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("run_id", str(run_id))
        active_spans[run_id] = span

    elif event_type == "on_chain_end" and run_id in active_spans:
        span = active_spans.pop(run_id)
        output = event.get("data", {}).get("output")
        span.set_attribute("output.keys", list(output.keys()) if isinstance(output, dict) else [])
        span.end()
```

> 这种方式不需要修改每个 Node 的代码，适合已有大量 Node 的场景。

---

## 5. 自定义 Node 级 Metric 上报思路

除了 Trace 之外，生产环境中通常还需要聚合指标（Metrics），如 Prometheus 或 StatsD。以下是针对 LangGraph 的 Node 级 Metric 设计思路。

### 5.1 建议采集的指标

| 指标名 | 类型 | 含义 |
|--------|------|------|
| `langgraph_node_execution_total` | Counter | 每个 Node 的执行总次数 |
| `langgraph_node_execution_duration_seconds` | Histogram | 每个 Node 的执行耗时分布 |
| `langgraph_node_errors_total` | Counter | 每个 Node 的异常次数 |
| `langgraph_retry_total` | Counter | 重试次数（按 Node 和原因分组） |
| `langgraph_tool_calls_total` | Counter | MCP/Tool 调用次数 |
| `langgraph_llm_tokens_total` | Counter | LLM 输入+输出 Token 数（可由事件流累加） |

### 5.2 实现方式：Node 装饰器

通过装饰器统一收集指标，不侵入业务逻辑：

```python
import time
from functools import wraps
from prometheus_client import Counter, Histogram

NODE_EXECUTION_TOTAL = Counter(
    "langgraph_node_execution_total",
    "Total number of node executions",
    ["node_name"]
)

NODE_EXECUTION_DURATION = Histogram(
    "langgraph_node_execution_duration_seconds",
    "Node execution duration in seconds",
    ["node_name"]
)

NODE_ERRORS_TOTAL = Counter(
    "langgraph_node_errors_total",
    "Total number of node execution errors",
    ["node_name", "error_type"]
)

def metric_node(func):
    """为 LangGraph Node 自动添加指标采集的装饰器"""
    node_name = func.__name__

    @wraps(func)
    async def async_wrapper(state):
        NODE_EXECUTION_TOTAL.labels(node_name=node_name).inc()
        start = time.perf_counter()
        try:
            result = await func(state)
            return result
        except Exception as e:
            NODE_ERRORS_TOTAL.labels(
                node_name=node_name,
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            NODE_EXECUTION_DURATION.labels(node_name=node_name).observe(duration)

    return async_wrapper

# 使用方式
@metric_node
async def _executor_node(state):
    # 原有逻辑...
    pass
```

### 5.3 结合 `orchestrator.py` 的扩展建议

在 `MultiAgentWorkflow` 中，可以在 `_build_workflow()` 阶段对所有 Node 统一包装：

```python
def _build_workflow(self):
    # 使用 metric_node 装饰器包装原始节点函数
    self.workflow.add_node("task_assigner", metric_node(self._task_assigner_node))
    self.workflow.add_node("tool_selector", metric_node(self._tool_selector_node))
    # ... 其他节点
```

这样做的好处是：
- Node 的业务逻辑和可观测性逻辑完全解耦；
- 后续切换 StatsD / OpenTelemetry Metrics 只需要修改装饰器内部实现；
- 不改变 `orchestrator.py` 的对外接口（`get_graph()` 返回值不变）。

---

## 小结

| 方案 | 适用场景 | 上手难度 | 项目中的对应文件 |
|------|---------|---------|-----------------|
| **LangSmith** | 快速搭建、调试 LLM 应用 | 低 | `examples/platform/langsmith_tracing.py` |
| **astream_events 日志** | 构建自研可观测平台、做审计 | 中 | `examples/advanced/streaming_output.py` |
| **OpenTelemetry 手动埋点** | 接入企业现有 APM、Jaeger | 中 | `examples/platform/opentelemetry_tracing.py` |
| **自定义 Metrics** | 生产环境监控、告警、SLO 管理 | 中 | 参考第 5 节的装饰器思路 |

建议学习路径：
1. 先用 **LangSmith** 跑通 Trace，熟悉 LangGraph 的执行链路；
2. 通过 **astream_events** 了解事件模型，为后续自研平台打基础；
3. 根据公司技术栈选择 **OpenTelemetry** 或 **Prometheus** 做深度集成。
