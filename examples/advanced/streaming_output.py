#!/usr/bin/env python3
"""
Streaming 流式输出示例

演示:
1. stream_mode="values"   — 每步输出完整状态
2. stream_mode="updates"  — 每步输出增量更新
3. stream_mode="messages" — 逐 token 输出消息 (聊天 UI 必备)
4. 多模式组合             — 同时输出 updates + messages
5. astream_events         — 细粒度事件流 (异步)

关键 API:
- app.stream(input, config, stream_mode="xxx")
- app.astream_events(input, config, version="v2")
- 多模式: stream_mode=["updates", "messages"]

LangGraph 概念: Streaming, Stream Modes, astream_events
"""

import os
import asyncio
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("未设置 OPENAI_API_KEY，使用模拟模式运行")
    USE_MOCK = True
else:
    USE_MOCK = False


# ============================================================
# 1. 定义 State 和节点
# ============================================================
class StreamState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


def research_node(state: StreamState) -> dict:
    """研究节点 — 模拟耗时研究"""
    query = state["messages"][-1].content if state["messages"] else "unknown"
    result = f"研究结果: 关于 '{query}' 的 3 个关键发现..."
    return {
        "messages": [AIMessage(content=result)],
        "summary": ""
    }


def summarize_node(state: StreamState) -> dict:
    """总结节点"""
    if USE_MOCK:
        summary = "这是模拟的总结内容，展示流式输出效果。"
        return {
            "messages": [AIMessage(content=summary)],
            "summary": summary
        }

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = llm.invoke(state["messages"] + [
        HumanMessage(content="请用 2-3 句话总结以上内容")
    ])
    return {
        "messages": [response],
        "summary": response.content
    }


def create_stream_graph():
    """创建用于演示流式输出的图"""
    graph = StateGraph(StreamState)
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


# ============================================================
# 2. 演示各种 stream mode
# ============================================================
def demo_stream_values():
    """stream_mode='values' — 每步输出完整状态"""
    print("=" * 60)
    print("演示 1: stream_mode='values' (每步完整状态)")
    print("=" * 60)

    app = create_stream_graph()
    input_data = {"messages": [HumanMessage(content="什么是 LangGraph？")]}

    for i, state in enumerate(app.stream(input_data, stream_mode="values")):
        msg_count = len(state.get("messages", []))
        summary = state.get("summary", "")
        print(f"  Step {i}: 消息数={msg_count}, summary='{summary[:30]}...' " if summary else f"  Step {i}: 消息数={msg_count}")

    print()


def demo_stream_updates():
    """stream_mode='updates' — 每步输出增量更新"""
    print("=" * 60)
    print("演示 2: stream_mode='updates' (每步增量更新)")
    print("=" * 60)

    app = create_stream_graph()
    input_data = {"messages": [HumanMessage(content="什么是 LangGraph？")]}

    for event in app.stream(input_data, stream_mode="updates"):
        for node_name, update in event.items():
            print(f"  [{node_name}] 更新的字段: {list(update.keys())}")
            if "messages" in update:
                last_msg = update["messages"][-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                print(f"    消息: {content[:60]}...")

    print()


def demo_stream_messages():
    """stream_mode='messages' — 逐消息输出 (聊天 UI 用)"""
    print("=" * 60)
    print("演示 3: stream_mode='messages' (逐消息输出)")
    print("=" * 60)

    app = create_stream_graph()
    input_data = {"messages": [HumanMessage(content="什么是 LangGraph？")]}

    print("  ", end="")
    for msg, metadata in app.stream(input_data, stream_mode="messages"):
        if hasattr(msg, "content") and msg.content:
            # 在真实 LLM 下，这里会逐 token 输出
            print(msg.content, end="", flush=True)
    print("\n")


def demo_stream_multi_mode():
    """多模式组合 — 同时获取 updates 和 messages"""
    print("=" * 60)
    print("演示 4: 多模式组合 stream_mode=['updates', 'messages']")
    print("=" * 60)

    app = create_stream_graph()
    input_data = {"messages": [HumanMessage(content="什么是 LangGraph？")]}

    for event in app.stream(input_data, stream_mode=["updates", "messages"]):
        mode, data = event
        if mode == "updates":
            for node_name in data:
                print(f"  [updates] 节点 '{node_name}' 完成")
        elif mode == "messages":
            msg, metadata = data
            if hasattr(msg, "content") and msg.content:
                print(f"  [messages] {msg.content[:50]}...")

    print()


# ============================================================
# 3. 异步演示: astream_events (最细粒度)
# ============================================================
async def demo_astream_events():
    """astream_events — 细粒度事件流"""
    print("=" * 60)
    print("演示 5: astream_events (细粒度内部事件)")
    print("=" * 60)

    app = create_stream_graph()
    input_data = {"messages": [HumanMessage(content="什么是 LangGraph？")]}

    event_count = 0
    async for event in app.astream_events(input_data, version="v2"):
        event_type = event["event"]

        # 只展示关键事件
        if event_type == "on_chain_start":
            name = event.get("name", "unknown")
            if name != "LangGraph":
                print(f"  🔵 节点开始: {name}")
        elif event_type == "on_chain_end":
            name = event.get("name", "unknown")
            if name != "LangGraph":
                print(f"  🟢 节点结束: {name}")
        elif event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)
                event_count += 1

    if event_count > 0:
        print()  # 换行
    print(f"\n  共捕获 {event_count} 个 token 事件")
    print()


# ============================================================
# 主函数
# ============================================================
def main():
    print("📡 Streaming 流式输出演示\n")

    demo_stream_values()
    demo_stream_updates()
    demo_stream_messages()
    demo_stream_multi_mode()

    # 异步演示
    asyncio.run(demo_astream_events())

    print("✅ 所有演示完成")


if __name__ == "__main__":
    main()
