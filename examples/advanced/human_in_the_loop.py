#!/usr/bin/env python3
"""
Human-in-the-Loop 示例

演示:
1. interrupt() 动态中断 — Agent 执行到关键步骤时暂停，等待人类确认
2. Command(resume=...) 恢复 — 人类提供输入后继续执行
3. 工具审批流 — 在执行危险工具前请求人类审批

关键 API:
- interrupt(value)           → 暂停图执行，返回 value 给调用方
- Command(resume=value)      → 恢复执行，value 成为 interrupt() 的返回值
- compile(interrupt_before=[]) → 静态断点 (在指定节点前暂停)

LangGraph 概念: Human-in-the-Loop, interrupt, Command, Breakpoints
"""

import os
from typing import TypedDict, Annotated, List, Optional
import operator
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()


# ============================================================
# 1. 定义 State
# ============================================================
class ApprovalState(TypedDict):
    """审批工作流状态"""
    messages: Annotated[list, add_messages]
    task: str
    risk_level: str           # low / medium / high
    approved: Optional[bool]  # 人类审批结果
    result: str


# ============================================================
# 2. 定义节点
# ============================================================
def analyze_task(state: ApprovalState) -> dict:
    """分析任务风险等级"""
    task = state["task"]

    # 简单规则判断风险等级
    high_risk_keywords = ["删除", "drop", "rm -rf", "格式化", "重置"]
    medium_risk_keywords = ["修改", "更新", "发送", "部署"]

    risk = "low"
    for kw in high_risk_keywords:
        if kw in task.lower():
            risk = "high"
            break
    if risk == "low":
        for kw in medium_risk_keywords:
            if kw in task.lower():
                risk = "medium"
                break

    print(f"  📋 任务: {task}")
    print(f"  ⚠️  风险等级: {risk}")

    return {
        "messages": [AIMessage(content=f"任务分析完成，风险等级: {risk}")],
        "risk_level": risk
    }


def human_approval(state: ApprovalState) -> dict:
    """
    人机交互节点 — 高风险任务需要人类审批

    interrupt() 会:
    1. 保存当前状态到 Checkpointer
    2. 暂停图执行
    3. 返回提示信息给调用方
    4. 等待 Command(resume=...) 恢复
    """
    risk = state["risk_level"]

    if risk == "low":
        # 低风险，自动通过
        print("  ✅ 低风险任务，自动批准")
        return {"approved": True}

    # 中/高风险 — 暂停等待人类审批
    prompt = f"⚠️ {risk.upper()} 风险操作: {state['task']}\n请回复 approve 或 reject"
    print(f"  ⏸️  等待人类审批...")

    # interrupt() 暂停执行
    # 当调用方用 Command(resume="approve") 恢复时，human_response = "approve"
    human_response = interrupt(prompt)

    approved = human_response.lower() in ["approve", "yes", "y", "确认", "批准"]
    print(f"  👤 人类决定: {'批准' if approved else '拒绝'}")

    return {
        "messages": [AIMessage(content=f"人类审批结果: {'批准' if approved else '拒绝'}")],
        "approved": approved
    }


def execute_task(state: ApprovalState) -> dict:
    """执行任务"""
    print(f"  🚀 执行任务: {state['task']}")
    return {
        "messages": [AIMessage(content=f"任务执行完成: {state['task']}")],
        "result": f"成功完成: {state['task']}"
    }


def reject_task(state: ApprovalState) -> dict:
    """拒绝任务"""
    print(f"  🚫 任务被拒绝")
    return {
        "messages": [AIMessage(content="任务已被拒绝")],
        "result": f"已拒绝: {state['task']}"
    }


# ============================================================
# 3. 路由函数
# ============================================================
def route_after_approval(state: ApprovalState) -> str:
    """根据审批结果路由"""
    if state.get("approved"):
        return "execute"
    return "reject"


# ============================================================
# 4. 构建图
# ============================================================
def create_approval_workflow():
    """创建审批工作流"""
    graph = StateGraph(ApprovalState)

    graph.add_node("analyze", analyze_task)
    graph.add_node("approval", human_approval)
    graph.add_node("execute", execute_task)
    graph.add_node("reject", reject_task)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "approval")
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "execute", "reject": "reject"}
    )
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    # 必须有 Checkpointer，interrupt() 才能保存状态
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 5. 演示
# ============================================================
def demo_low_risk():
    """演示低风险任务 — 自动通过"""
    print("=" * 60)
    print("演示 1: 低风险任务 (自动批准，不中断)")
    print("=" * 60)

    app = create_approval_workflow()
    config = {"configurable": {"thread_id": "demo-low-risk"}}

    result = app.invoke(
        {"task": "查询用户列表", "messages": []},
        config=config
    )
    print(f"  结果: {result['result']}\n")


def demo_high_risk():
    """演示高风险任务 — 需要人类审批"""
    print("=" * 60)
    print("演示 2: 高风险任务 (interrupt 等待审批)")
    print("=" * 60)

    app = create_approval_workflow()
    config = {"configurable": {"thread_id": "demo-high-risk"}}

    # 第一次调用 — 会在 interrupt() 处暂停
    print("\n[第一次调用] 提交高风险任务:")
    result = app.invoke(
        {"task": "删除所有用户数据", "messages": []},
        config=config
    )
    # 图暂停了，result 是中断时的状态
    print(f"  图已暂停，等待人类输入")

    # 模拟人类审批 — 批准
    print("\n[第二次调用] 人类批准:")
    result = app.invoke(
        Command(resume="approve"),
        config=config
    )
    print(f"  结果: {result['result']}\n")


def demo_high_risk_reject():
    """演示高风险任务 — 人类拒绝"""
    print("=" * 60)
    print("演示 3: 高风险任务 (人类拒绝)")
    print("=" * 60)

    app = create_approval_workflow()
    config = {"configurable": {"thread_id": "demo-reject"}}

    # 提交任务
    print("\n[第一次调用] 提交高风险任务:")
    app.invoke(
        {"task": "rm -rf /important_data", "messages": []},
        config=config
    )
    print(f"  图已暂停")

    # 人类拒绝
    print("\n[第二次调用] 人类拒绝:")
    result = app.invoke(
        Command(resume="reject"),
        config=config
    )
    print(f"  结果: {result['result']}\n")


# ============================================================
# 主函数
# ============================================================
def main():
    print("🤝 Human-in-the-Loop 审批流演示\n")
    demo_low_risk()
    demo_high_risk()
    demo_high_risk_reject()
    print("✅ 所有演示完成")


if __name__ == "__main__":
    main()
