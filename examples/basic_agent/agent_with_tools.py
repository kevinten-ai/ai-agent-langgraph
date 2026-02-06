#!/usr/bin/env python3
"""
带工具的Agent示例

这个示例展示了如何创建一个能够使用工具的Agent。
Agent可以调用计算器工具和网络搜索工具来回答复杂问题。
"""

import os
import requests
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

# 检查API密钥
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class AgentWithToolsState(TypedDict):
    """带工具的Agent状态"""
    messages: Annotated[List[dict], operator.add]  # 对话历史
    user_input: str                               # 用户输入
    current_tool_calls: List[dict]                # 当前的工具调用
    final_answer: str                            # 最终答案


# 定义工具
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        # 注意：这里使用eval仅用于演示，生产环境中应该使用更安全的方法
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    简单的网络搜索工具（模拟）

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果摘要
    """
    try:
        # 这里使用一个简单的模拟搜索
        # 实际项目中应该使用真实的搜索API如Google、Bing等

        # 模拟搜索结果
        mock_results = {
            "python": "Python是一种高级编程语言，以其简单性和可读性而闻名。",
            "ai agent": "AI Agent是能够自主执行任务的智能系统。",
            "machine learning": "机器学习是人工智能的一个子领域。",
            "langgraph": "LangGraph是一个用于构建AI Agent的框架。"
        }

        # 查找相关结果
        results = []
        query_lower = query.lower()

        for key, value in mock_results.items():
            if key in query_lower or query_lower in key:
                results.append(f"- {key.title()}: {value}")

        if results:
            return f"搜索结果:\n" + "\n".join(results[:max_results])
        else:
            return f"为 '{query}' 找到以下相关信息：这是一个很好的问题，但我的知识库中没有具体答案。"

    except Exception as e:
        return f"搜索错误: {str(e)}"


def process_user_input(state: AgentWithToolsState) -> AgentWithToolsState:
    """
    处理用户输入

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    human_message = HumanMessage(content=state["user_input"])

    return {
        "messages": [human_message],
        "user_input": state["user_input"],
        "current_tool_calls": [],
        "final_answer": ""
    }


def decide_next_action(state: AgentWithToolsState) -> AgentWithToolsState:
    """
    决定下一步行动：使用工具还是直接回复

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    # 绑定工具
    llm_with_tools = llm.bind_tools([calculator, web_search])

    # 获取AI的响应
    response = llm_with_tools.invoke(state["messages"])

    # 检查是否有工具调用
    tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_calls = response.tool_calls

    return {
        "messages": [response],
        "user_input": state["user_input"],
        "current_tool_calls": tool_calls,
        "final_answer": ""
    }


def execute_tools(state: AgentWithToolsState) -> AgentWithToolsState:
    """
    执行工具调用

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    tool_messages = []

    for tool_call in state["current_tool_calls"]:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        try:
            if tool_name == "calculator":
                result = calculator.invoke(tool_args)
            elif tool_name == "web_search":
                result = web_search.invoke(tool_args)
            else:
                result = f"未知工具: {tool_name}"

            # 创建工具消息
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

        except Exception as e:
            tool_message = ToolMessage(
                content=f"工具执行错误: {str(e)}",
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

    return {
        "messages": tool_messages,
        "user_input": state["user_input"],
        "current_tool_calls": [],
        "final_answer": ""
    }


def generate_final_answer(state: AgentWithToolsState) -> AgentWithToolsState:
    """
    生成最终答案

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    # 使用完整的对话历史生成最终答案
    response = llm.invoke(state["messages"])

    response_text = response.content if hasattr(response, 'content') else str(response)

    return {
        "messages": [response],
        "user_input": state["user_input"],
        "current_tool_calls": [],
        "final_answer": response_text
    }


def should_use_tools(state: AgentWithToolsState) -> str:
    """
    判断是否需要使用工具

    Args:
        state: 当前状态

    Returns:
        下一个节点名称
    """
    if state["current_tool_calls"]:
        return "execute_tools"
    else:
        return "generate_final_answer"


def create_agent_with_tools_graph():
    """
    创建带工具的Agent图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(AgentWithToolsState)

    # 添加节点
    graph.add_node("process_input", process_user_input)
    graph.add_node("decide_action", decide_next_action)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("generate_final_answer", generate_final_answer)

    # 定义边
    graph.add_edge(START, "process_input")
    graph.add_edge("process_input", "decide_action")

    # 条件边：根据是否有工具调用决定下一步
    graph.add_conditional_edges(
        "decide_action",
        should_use_tools,
        {
            "execute_tools": "execute_tools",
            "generate_final_answer": "generate_final_answer"
        }
    )

    graph.add_edge("execute_tools", "generate_final_answer")
    graph.add_edge("generate_final_answer", END)

    return graph.compile()


def chat_with_agent():
    """
    与带工具的Agent交互
    """
    print("🛠️ 欢迎使用智能工具Agent！")
    print("我可以帮你进行数学计算和信息搜索。")
    print("输入 'quit' 退出对话")
    print("-" * 50)

    agent = create_agent_with_tools_graph()
    conversation_history = []

    while True:
        user_input = input("你: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("🛠️ 再见！")
            break

        if not user_input:
            print("🛠️ 请输入有效的问题...")
            continue

        try:
            result = agent.invoke({
                "messages": conversation_history,
                "user_input": user_input,
                "current_tool_calls": [],
                "final_answer": ""
            })

            print(f"🛠️: {result['final_answer']}")

            # 更新对话历史
            conversation_history.extend(result["messages"])

            # 限制历史长度
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        chat_with_agent()
    except KeyboardInterrupt:
        print("\n🛠️ 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



