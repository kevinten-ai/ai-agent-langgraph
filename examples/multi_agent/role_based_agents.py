#!/usr/bin/env python3
"""
角色分工的Multi-Agent示例

这个示例展示了如何创建多个具有不同角色的Agent，
它们通过消息传递和协作来完成复杂任务。
"""

import os
from typing import TypedDict, Annotated, List, Literal
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class MultiAgentState(TypedDict):
    """多Agent协作状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 全局消息历史
    current_agent: str                                   # 当前活跃的Agent
    task: str                                            # 原始任务
    research_result: str                                 # 研究结果
    analysis_result: str                                 # 分析结果
    final_report: str                                    # 最终报告
    next_step: str                                       # 下一步骤


class ResearchAgent:
    """研究Agent - 负责收集信息"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        )

    def research(self, task: str) -> str:
        """执行研究任务"""
        prompt = f"""你是一个专业的研究员。你的任务是收集和整理关于以下主题的信息：

任务: {task}

请提供：
1. 相关背景信息
2. 关键事实和数据
3. 重要趋势和发展
4. 相关资源和参考资料

请保持客观和全面。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class AnalysisAgent:
    """分析Agent - 负责分析数据和提供洞察"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

    def analyze(self, task: str, research_data: str) -> str:
        """执行分析任务"""
        prompt = f"""你是一个专业的数据分析师。基于以下研究数据，分析任务主题：

原始任务: {task}

研究数据:
{research_data}

请提供：
1. 数据关键发现
2. 趋势分析
3. 潜在影响和含义
4. 建议和结论

请用数据支持你的分析。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class ReportAgent:
    """报告Agent - 负责生成最终报告"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )

    def generate_report(self, task: str, research_data: str, analysis_data: str) -> str:
        """生成最终报告"""
        prompt = f"""你是一个专业的报告撰写者。基于研究和分析结果，生成一份完整的报告：

原始任务: {task}

研究结果:
{research_data}

分析结果:
{analysis_data}

请生成一份结构化的报告，包括：
1. 执行摘要
2. 背景介绍
3. 主要发现
4. 分析洞察
5. 结论和建议
6. 参考资料

报告应该专业、清晰、有说服力。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


def coordinator_agent(state: MultiAgentState) -> MultiAgentState:
    """
    协调Agent - 决定任务分配和流程控制

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    if not state.get("research_result"):
        # 第一步：分配研究任务
        return {
            "messages": [AIMessage(content="开始执行研究任务...")],
            "current_agent": "researcher",
            "next_step": "research"
        }
    elif not state.get("analysis_result"):
        # 第二步：分配分析任务
        return {
            "messages": [AIMessage(content="开始执行分析任务...")],
            "current_agent": "analyst",
            "next_step": "analyze"
        }
    else:
        # 第三步：分配报告生成任务
        return {
            "messages": [AIMessage(content="开始生成最终报告...")],
            "current_agent": "reporter",
            "next_step": "report"
        }


def research_agent_node(state: MultiAgentState) -> MultiAgentState:
    """研究Agent节点"""
    researcher = ResearchAgent()
    research_result = researcher.research(state["task"])

    return {
        "messages": [AIMessage(content=f"研究完成：{research_result[:100]}...")],
        "research_result": research_result,
        "current_agent": "analyst"
    }


def analysis_agent_node(state: MultiAgentState) -> MultiAgentState:
    """分析Agent节点"""
    analyst = AnalysisAgent()
    analysis_result = analyst.analyze(state["task"], state["research_result"])

    return {
        "messages": [AIMessage(content=f"分析完成：{analysis_result[:100]}...")],
        "analysis_result": analysis_result,
        "current_agent": "reporter"
    }


def report_agent_node(state: MultiAgentState) -> MultiAgentState:
    """报告Agent节点"""
    reporter = ReportAgent()
    final_report = reporter.generate_report(
        state["task"],
        state["research_result"],
        state["analysis_result"]
    )

    return {
        "messages": [AIMessage(content="最终报告已生成")],
        "final_report": final_report,
        "current_agent": "completed"
    }


def should_continue(state: MultiAgentState) -> str:
    """
    判断是否继续协作

    Args:
        state: 当前状态

    Returns:
        下一个节点
    """
    if not state.get("research_result"):
        return "research"
    elif not state.get("analysis_result"):
        return "analyze"
    elif not state.get("final_report"):
        return "report"
    else:
        return END


def create_multi_agent_graph():
    """
    创建多Agent协作图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(MultiAgentState)

    # 添加节点
    graph.add_node("coordinator", coordinator_agent)
    graph.add_node("research", research_agent_node)
    graph.add_node("analyze", analysis_agent_node)
    graph.add_node("report", report_agent_node)

    # 定义边
    graph.add_edge(START, "coordinator")

    # 条件边：根据任务完成情况决定下一步
    graph.add_conditional_edges(
        "coordinator",
        should_continue,
        {
            "research": "research",
            "analyze": "analyze",
            "report": "report",
            END: END
        }
    )

    # 各Agent完成后的反馈循环
    graph.add_edge("research", "coordinator")
    graph.add_edge("analyze", "coordinator")
    graph.add_edge("report", END)

    return graph.compile()


def run_multi_agent_collaboration():
    """
    运行多Agent协作示例
    """
    print("🤝 多Agent协作系统启动")
    print("这个系统包含三个专业Agent：研究员、分析师、报告撰写者")
    print("-" * 60)

    # 创建多Agent系统
    multi_agent_system = create_multi_agent_graph()

    # 示例任务
    sample_tasks = [
        "分析人工智能在医疗保健领域的应用现状和发展趋势",
        "评估区块链技术在金融行业的潜在影响",
        "研究远程工作对员工生产力和幸福感的影响"
    ]

    print("可用的示例任务：")
    for i, task in enumerate(sample_tasks, 1):
        print(f"{i}. {task}")

    print("\n请选择任务编号，或输入自定义任务：")

    while True:
        user_input = input("\n请选择 (1-3) 或输入自定义任务: ").strip()

        if user_input in ['1', '2', '3']:
            task = sample_tasks[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("🤝 再见！")
            break
        elif user_input:
            task = user_input
        else:
            print("❌ 请输入有效任务")
            continue

        print(f"\n🔄 开始处理任务: {task}")
        print("=" * 60)

        try:
            # 执行多Agent协作
            result = multi_agent_system.invoke({
                "messages": [HumanMessage(content=f"任务：{task}")],
                "current_agent": "coordinator",
                "task": task,
                "research_result": "",
                "analysis_result": "",
                "final_report": "",
                "next_step": "research"
            })

            # 显示结果
            print("\n📊 执行过程:")
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"🤖 {msg.content}")

            print("\n📋 最终报告:")
            print("-" * 40)
            print(result["final_report"])

            print("\n✅ 任务完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")
            print("请检查网络连接和API配置")


def main():
    """主函数"""
    try:
        run_multi_agent_collaboration()
    except KeyboardInterrupt:
        print("\n🤝 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



