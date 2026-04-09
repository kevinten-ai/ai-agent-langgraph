#!/usr/bin/env python3
"""
OpenTelemetry 手动埋点示例

演示如何使用 opentelemetry-api/sdk 给 LangGraph 节点手动添加 span，
并通过 ConsoleSpanExporter 将 trace 输出到控制台。

运行方式:
    python examples/platform/opentelemetry_tracing.py

说明:
- 不需要外部 Jaeger/Collector，直接看控制台输出即可。
- 使用一个极简的 LangGraph 图，在 Node 内部手动 start/end span。
"""

import os
import sys
import asyncio
import time
from typing import TypedDict

# 将项目根目录加入 PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ============================================================
# 1. 初始化 OpenTelemetry TracerProvider 和 ConsoleSpanExporter
# ============================================================
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter  # noqa: E402
from opentelemetry.trace.status import Status, StatusCode  # noqa: E402

# 创建 Provider 并添加控制台导出器
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 获取 tracer，名称通常对应模块/服务名
tracer = trace.get_tracer("ai-agent-langgraph.opentelemetry_demo")

# ============================================================
# 2. 定义一个极简的图和 State
# ============================================================
from langgraph.graph import StateGraph, START, END  # noqa: E402


class DemoState(TypedDict):
    """演示用状态，仅包含一个输入和一个累计结果。"""
    query: str
    result: str


async def node_research(state: DemoState) -> DemoState:
    """
    研究节点：模拟耗时操作，并用 tracer 手动创建 span。
    """
    # start_as_current_span 会自动将当前 span 放入上下文，
    # 后续如果有嵌套 span，可以自动建立父子关系。
    with tracer.start_as_current_span("node.research") as span:
        span.set_attribute("node.type", "research")
        span.set_attribute("input.query", state.get("query", ""))

        try:
            # 模拟耗时研究
            await asyncio.sleep(0.2)
            output = f"关于 '{state['query']}' 的研究结果已生成。"

            span.set_attribute("output.length", len(output))
            span.set_attribute("output.status", "success")
            return {"query": state["query"], "result": output}
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def node_summarize(state: DemoState) -> DemoState:
    """
    总结节点：模拟 LLM 总结过程，并手动创建 span。
    """
    # 也可以不通过上下文管理器，而是显式 start_span / end，
    # 但需要手动设置 parent context。这里仍使用上下文管理器简化代码。
    with tracer.start_as_current_span("node.summarize") as span:
        span.set_attribute("node.type", "summarize")
        span.set_attribute("input.result_preview", state.get("result", "")[:50])

        try:
            # 模拟 LLM 调用耗时
            await asyncio.sleep(0.15)
            summary = f"总结：{state['result'][:30]}..."

            span.set_attribute("output.length", len(summary))
            span.set_attribute("output.status", "success")
            return {"query": state["query"], "result": summary}
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def build_demo_graph():
    """构造并编译演示图。"""
    builder = StateGraph(DemoState)
    builder.add_node("research", node_research)
    builder.add_node("summarize", node_summarize)
    builder.add_edge(START, "research")
    builder.add_edge("research", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


# ============================================================
# 3. 基于 astream_events 实现无侵入式 OTel 集成
# ============================================================
async def run_with_event_spans(app, input_state: DemoState):
    """
    高级用法：不修改 node 函数，而是在 astream_events 循环中
    根据 on_chain_start / on_chain_end 自动开启和关闭 span。
    """
    from opentelemetry.trace import SpanKind  # noqa: E402

    active_spans: dict = {}
    final_result = None

    async for event in app.astream_events(input_state, version="v2"):
        event_type = event["event"]
        name = event.get("name", "unknown")
        run_id = event.get("run_id")

        # LangGraph 自身图调用也会发出 on_chain_start/end，name 为 "LangGraph"
        # 这里过滤掉它，只关注业务节点。
        if event_type == "on_chain_start" and name not in ("LangGraph", ""):
            span = tracer.start_span(
                name=f"node.{name}",
                kind=SpanKind.INTERNAL,
            )
            span.set_attribute("run_id", str(run_id))
            span.set_attribute("event_source", "astream_events")
            active_spans[run_id] = span

        elif event_type == "on_chain_end" and run_id in active_spans:
            span = active_spans.pop(run_id)
            data = event.get("data", {})
            output = data.get("output")
            if isinstance(output, dict):
                span.set_attribute("output.keys", list(output.keys()))
            span.end()

        # 收集最终结果（values 模式会在 on_chain_end 里拿到完整状态）
        if event_type == "on_chain_end" and name == "LangGraph":
            data = event.get("data", {})
            final_result = data.get("output")

    return final_result


# ============================================================
# 4. 主函数
# ============================================================
async def main():
    print("=" * 60)
    print("OpenTelemetry Tracing 演示")
    print("=" * 60)

    # ---------------- 演示 A：Node 内手动埋点 ----------------
    print("\n[演示 A] Node 内手动 start_as_current_span")
    app_a = build_demo_graph()
    result_a = await app_a.ainvoke({"query": "LangGraph 可观测性", "result": ""})
    print(f"  执行结果: {result_a}")

    # 给控制台导出器一点时间 flush
    await asyncio.sleep(0.5)

    # ---------------- 演示 B：基于事件流自动埋点 ----------------
    print("\n[演示 B] 基于 astream_events 自动创建/结束 span")
    app_b = build_demo_graph()
    result_b = await run_with_event_spans(app_b, {"query": "OpenTelemetry 集成", "result": ""})
    print(f"  执行结果: {result_b}")

    # 等待 span 输出到控制台
    await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print("✅ 演示完成。请查看上方的 ConsoleSpanExporter 输出。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
