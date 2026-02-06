#!/usr/bin/env python3
"""
简单聊天机器人示例

这个示例展示了如何使用LangGraph创建一个基本的问答Agent。
Agent能够接收用户输入，调用LLM生成回复，并维护对话历史。
"""

import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

# 检查API密钥
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class ChatbotState(TypedDict):
    """聊天机器人状态"""
    messages: Annotated[List[dict], operator.add]  # 对话历史
    user_input: str                                # 用户输入
    response: str                                  # AI回复


def process_input(state: ChatbotState) -> ChatbotState:
    """
    处理用户输入，将其添加到消息历史中

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    # 将用户输入转换为消息格式
    human_message = HumanMessage(content=state["user_input"])

    # 返回更新后的状态
    return {
        "messages": [human_message],
        "user_input": state["user_input"],
        "response": ""
    }


def generate_response(state: ChatbotState) -> ChatbotState:
    """
    使用LLM生成回复

    Args:
        state: 当前状态

    Returns:
        包含AI回复的状态
    """
    # 初始化LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500
    )

    # 调用LLM生成回复
    ai_response = llm.invoke(state["messages"])

    # 确保回复是字符串格式
    response_text = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

    # 返回更新后的状态
    return {
        "messages": [ai_response],
        "user_input": state["user_input"],
        "response": response_text
    }


def create_chatbot_graph():
    """
    创建聊天机器人图

    Returns:
        编译后的状态图
    """
    # 创建状态图
    graph = StateGraph(ChatbotState)

    # 添加节点
    graph.add_node("process_input", process_input)
    graph.add_node("generate_response", generate_response)

    # 定义边
    graph.add_edge(START, "process_input")
    graph.add_edge("process_input", "generate_response")
    graph.add_edge("generate_response", END)

    # 编译图
    return graph.compile()


def chat_with_bot():
    """
    与聊天机器人交互的函数
    """
    print("🤖 欢迎使用简单聊天机器人！")
    print("输入 'quit' 退出对话")
    print("-" * 50)

    # 创建聊天机器人
    chatbot = create_chatbot_graph()

    # 初始化对话历史
    conversation_history = []

    while True:
        # 获取用户输入
        user_input = input("你: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("🤖 再见！")
            break

        if not user_input:
            print("🤖 请输入有效的问题...")
            continue

        try:
            # 调用聊天机器人
            result = chatbot.invoke({
                "messages": conversation_history,
                "user_input": user_input,
                "response": ""
            })

            # 显示回复
            print(f"🤖: {result['response']}")

            # 更新对话历史
            conversation_history.extend(result["messages"])

            # 限制历史长度，避免token过多
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print("请检查网络连接和API密钥配置")


def main():
    """主函数"""
    try:
        chat_with_bot()
    except KeyboardInterrupt:
        print("\n🤖 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



