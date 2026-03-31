#!/usr/bin/env python3
"""
Swarm 多 Agent 自主交接示例

演示:
1. Command(goto=...) 实现 Agent 间 handoff
2. 手动构建 Swarm — Agent 自主决定交给谁
3. 使用 langgraph-swarm 库 (如果安装了)

关键 API:
- Command(update={...}, goto="target_node")  → 更新状态 + 跳转
- create_handoff_tool(agent_name="xxx")       → 创建交接工具 (langgraph-swarm)
- create_swarm(agents=[...])                  → 创建 Swarm (langgraph-swarm)

LangGraph 概念: Command, Handoff, Swarm Pattern, Multi-Agent
"""

from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command


# ============================================================
# 1. 定义 State
# ============================================================
class SwarmState(TypedDict):
    """Swarm 状态"""
    messages: Annotated[list, add_messages]
    current_agent: str               # 当前活跃的 Agent
    task: str                        # 原始任务
    handoff_count: int               # 交接次数 (防无限循环)
    result: str


# ============================================================
# 2. 定义 Agent 节点 (使用 Command 实现 handoff)
# ============================================================
def math_agent(state: SwarmState) -> Command:
    """
    数学专家 Agent

    处理数学问题；如果不是数学问题，handoff 给 writer
    返回 Command 同时更新状态和控制流向
    """
    task = state["task"].lower()
    handoff_count = state.get("handoff_count", 0)

    # 防止无限循环
    if handoff_count >= 3:
        return Command(
            update={
                "messages": [AIMessage(content="[math] 已达最大交接次数，直接处理")],
                "result": f"[math] 强制处理: {state['task']}"
            },
            goto=END
        )

    # 判断是否是自己能处理的任务
    math_keywords = ["计算", "数学", "加", "减", "乘", "除", "方程", "公式", "+", "-", "*", "/"]
    is_math = any(kw in task for kw in math_keywords)

    if is_math:
        # 自己处理
        print(f"  🔢 [math_agent] 处理数学任务: {state['task']}")
        return Command(
            update={
                "messages": [AIMessage(content=f"[math] 计算完成: {state['task']}")],
                "current_agent": "math",
                "result": f"数学计算结果: {state['task']} = [已计算]"
            },
            goto=END
        )
    else:
        # Handoff 给 writer
        print(f"  🔢 [math_agent] 这不是数学任务，交接给 writer_agent")
        return Command(
            update={
                "messages": [AIMessage(content="[math] 这不是数学问题，交给写作专家")],
                "current_agent": "writer",
                "handoff_count": handoff_count + 1
            },
            goto="writer_agent"  # 关键: Command(goto=...) 实现 handoff
        )


def writer_agent(state: SwarmState) -> Command:
    """
    写作专家 Agent

    处理写作任务；如果是数据分析，handoff 给 analyst
    """
    task = state["task"].lower()
    handoff_count = state.get("handoff_count", 0)

    if handoff_count >= 3:
        return Command(
            update={
                "messages": [AIMessage(content="[writer] 已达最大交接次数，直接处理")],
                "result": f"[writer] 强制处理: {state['task']}"
            },
            goto=END
        )

    write_keywords = ["写", "文章", "报告", "故事", "总结", "文档"]
    is_writing = any(kw in task for kw in write_keywords)

    if is_writing:
        print(f"  ✍️  [writer_agent] 处理写作任务: {state['task']}")
        return Command(
            update={
                "messages": [AIMessage(content=f"[writer] 写作完成: {state['task']}")],
                "current_agent": "writer",
                "result": f"写作成果: 关于 '{state['task']}' 的文章已完成"
            },
            goto=END
        )
    else:
        print(f"  ✍️  [writer_agent] 这不是写作任务，交接给 analyst_agent")
        return Command(
            update={
                "messages": [AIMessage(content="[writer] 这不是写作问题，交给分析专家")],
                "current_agent": "analyst",
                "handoff_count": handoff_count + 1
            },
            goto="analyst_agent"
        )


def analyst_agent(state: SwarmState) -> Command:
    """
    分析专家 Agent

    处理分析任务；如果是计算，handoff 给 math
    """
    task = state["task"].lower()
    handoff_count = state.get("handoff_count", 0)

    if handoff_count >= 3:
        return Command(
            update={
                "messages": [AIMessage(content="[analyst] 已达最大交接次数，直接处理")],
                "result": f"[analyst] 强制处理: {state['task']}"
            },
            goto=END
        )

    analysis_keywords = ["分析", "数据", "趋势", "统计", "评估", "调研"]
    is_analysis = any(kw in task for kw in analysis_keywords)

    if is_analysis:
        print(f"  📊 [analyst_agent] 处理分析任务: {state['task']}")
        return Command(
            update={
                "messages": [AIMessage(content=f"[analyst] 分析完成: {state['task']}")],
                "current_agent": "analyst",
                "result": f"分析报告: '{state['task']}' 的深度分析已完成"
            },
            goto=END
        )
    else:
        print(f"  📊 [analyst_agent] 这不是分析任务，交接给 math_agent")
        return Command(
            update={
                "messages": [AIMessage(content="[analyst] 不是分析问题，交给数学专家")],
                "current_agent": "math",
                "handoff_count": handoff_count + 1
            },
            goto="math_agent"
        )


# ============================================================
# 3. 路由器 — 根据 current_agent 决定入口
# ============================================================
def router(state: SwarmState) -> Command:
    """初始路由 — 根据任务内容决定第一个 Agent"""
    task = state["task"].lower()

    math_keywords = ["计算", "数学", "+", "-", "*", "/"]
    write_keywords = ["写", "文章", "报告", "故事"]

    if any(kw in task for kw in math_keywords):
        target = "math_agent"
    elif any(kw in task for kw in write_keywords):
        target = "writer_agent"
    else:
        target = "analyst_agent"

    print(f"  🚦 [router] 任务 '{state['task']}' → 分配给 {target}")
    return Command(goto=target)


# ============================================================
# 4. 构建 Swarm 图
# ============================================================
def create_swarm_graph():
    """构建手动 Swarm"""
    graph = StateGraph(SwarmState)

    graph.add_node("router", router)
    graph.add_node("math_agent", math_agent)
    graph.add_node("writer_agent", writer_agent)
    graph.add_node("analyst_agent", analyst_agent)

    graph.add_edge(START, "router")
    # 注意: 不需要 add_edge 连接 Agent 之间
    # 因为 Command(goto=...) 在运行时动态决定流向

    return graph.compile()


# ============================================================
# 5. 演示
# ============================================================
def demo_swarm():
    """演示 Swarm 交接"""
    app = create_swarm_graph()

    test_tasks = [
        ("计算 123 + 456", "应该由 math_agent 直接处理"),
        ("写一篇关于 AI 的文章", "应该由 writer_agent 直接处理"),
        ("分析市场趋势数据", "应该由 analyst_agent 直接处理"),
        ("帮我整理下笔记", "router → analyst → writer (交接链)"),
    ]

    for task, expected in test_tasks:
        print("=" * 60)
        print(f"任务: {task}")
        print(f"预期: {expected}")
        print("-" * 40)

        result = app.invoke({
            "messages": [HumanMessage(content=task)],
            "current_agent": "",
            "task": task,
            "handoff_count": 0,
            "result": ""
        })

        print(f"  结果: {result['result']}")
        print(f"  最终 Agent: {result.get('current_agent', 'N/A')}")
        print(f"  交接次数: {result.get('handoff_count', 0)}")
        print()


# ============================================================
# 主函数
# ============================================================
def main():
    print("🐝 Swarm 多 Agent 自主交接演示\n")
    print("三个 Agent: math_agent, writer_agent, analyst_agent")
    print("它们根据任务内容自主决定处理还是交接\n")
    demo_swarm()
    print("✅ 所有演示完成")


if __name__ == "__main__":
    main()
