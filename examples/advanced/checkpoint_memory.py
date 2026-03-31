#!/usr/bin/env python3
"""
Checkpoint 与多轮记忆示例

演示:
1. MemorySaver 实现跨轮对话记忆 (同一 thread_id 保持上下文)
2. 不同 thread_id 之间完全隔离
3. 时间旅行 (get_state_history) 查看和回滚历史状态

关键 API:
- MemorySaver()                              → 内存 Checkpointer
- graph.compile(checkpointer=checkpointer)   → 编译时注入
- config = {"configurable": {"thread_id": "xxx"}}  → 线程隔离
- app.get_state(config)                      → 获取当前状态
- app.get_state_history(config)              → 获取历史快照
- app.invoke(None, target_config)            → 从历史状态恢复执行

LangGraph 概念: Checkpointer, Thread, Time Travel
"""

import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("未设置 OPENAI_API_KEY，使用模拟模式运行")
    USE_MOCK = True
else:
    USE_MOCK = False


# ============================================================
# 1. 定义 State
# ============================================================
class ChatState(TypedDict):
    """对话状态 — add_messages 归约器自动管理消息列表"""
    messages: Annotated[list, add_messages]


# ============================================================
# 2. 定义节点
# ============================================================
def chatbot(state: ChatState) -> dict:
    """聊天节点 — 调用 LLM 生成回复"""
    if USE_MOCK:
        # 模拟模式：不调用 API
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "名字" in last_msg or "叫什么" in last_msg:
            reply = "根据之前的对话，你告诉我你的名字了。让我回忆一下..."
            # 检查历史消息中是否有名字
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage) and "我叫" in msg.content:
                    name = msg.content.replace("我叫", "").strip()
                    reply = f"你叫{name}！"
                    break
        elif "我叫" in last_msg:
            name = last_msg.replace("我叫", "").strip()
            reply = f"你好，{name}！很高兴认识你。"
        else:
            reply = f"[模拟回复] 收到：{last_msg}"
        return {"messages": [AIMessage(content=reply)]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 3. 构建图 + Checkpointer
# ============================================================
def create_chat_with_memory():
    """创建带记忆的聊天图"""
    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # 关键: 注入 MemorySaver
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 4. 演示: 多轮记忆
# ============================================================
def demo_multi_turn_memory():
    """演示同一 thread 内的多轮记忆"""
    print("=" * 60)
    print("演示 1: 多轮对话记忆 (同一 thread_id)")
    print("=" * 60)

    app = create_chat_with_memory()
    config = {"configurable": {"thread_id": "user-alice"}}

    # 第一轮
    result1 = app.invoke(
        {"messages": [HumanMessage(content="我叫张三")]},
        config=config
    )
    print(f"用户: 我叫张三")
    print(f"AI:   {result1['messages'][-1].content}")

    # 第二轮 — Checkpointer 自动恢复上下文
    result2 = app.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？")]},
        config=config
    )
    print(f"\n用户: 我叫什么名字？")
    print(f"AI:   {result2['messages'][-1].content}")
    print(f"\n消息总数: {len(result2['messages'])} (自动累积)")


# ============================================================
# 5. 演示: Thread 隔离
# ============================================================
def demo_thread_isolation():
    """演示不同 thread 之间的隔离"""
    print("\n" + "=" * 60)
    print("演示 2: Thread 隔离 (不同 thread_id)")
    print("=" * 60)

    app = create_chat_with_memory()

    # Thread A
    config_a = {"configurable": {"thread_id": "thread-A"}}
    app.invoke(
        {"messages": [HumanMessage(content="我叫Alice")]},
        config=config_a
    )
    print("Thread A: 用户说 '我叫Alice'")

    # Thread B — 完全独立
    config_b = {"configurable": {"thread_id": "thread-B"}}
    result_b = app.invoke(
        {"messages": [HumanMessage(content="我叫什么？")]},
        config=config_b
    )
    print(f"Thread B: 用户问 '我叫什么？'")
    print(f"Thread B: AI 回复 '{result_b['messages'][-1].content}'")
    print("→ Thread B 不知道 Thread A 的信息，隔离成功")


# ============================================================
# 6. 演示: 时间旅行
# ============================================================
def demo_time_travel():
    """演示时间旅行 — 查看和回滚历史状态"""
    print("\n" + "=" * 60)
    print("演示 3: 时间旅行 (get_state_history)")
    print("=" * 60)

    app = create_chat_with_memory()
    config = {"configurable": {"thread_id": "time-travel-demo"}}

    # 执行 3 轮对话
    conversations = ["你好", "今天天气怎么样", "帮我写首诗"]
    for msg in conversations:
        app.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        print(f"  完成: '{msg}'")

    # 查看历史
    print("\n历史状态快照:")
    history = list(app.get_state_history(config))
    for i, state in enumerate(history):
        msg_count = len(state.values.get("messages", []))
        step = state.metadata.get("step", "?")
        print(f"  快照 {i}: step={step}, 消息数={msg_count}")

    # 回滚到第一轮之后的状态
    if len(history) >= 4:
        target = history[-3]  # 较早的状态
        print(f"\n回滚到快照 (消息数={len(target.values.get('messages', []))})")
        result = app.invoke(
            {"messages": [HumanMessage(content="从这里继续")]},
            target.config
        )
        print(f"从历史状态恢复后 AI: {result['messages'][-1].content}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("🧠 Checkpoint 与多轮记忆演示\n")
    demo_multi_turn_memory()
    demo_thread_isolation()
    demo_time_travel()
    print("\n✅ 所有演示完成")


if __name__ == "__main__":
    main()
