#!/usr/bin/env python3
"""
条件分支工作流示例

这个示例展示了如何在LangGraph中实现条件分支逻辑，
根据不同的条件执行不同的处理路径。
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


class ConditionalWorkflowState(TypedDict):
    """条件分支工作流状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史
    user_query: str                                      # 用户查询
    query_type: str                                       # 查询类型: 'factual', 'analytical', 'creative', 'unknown'
    complexity_level: str                                 # 复杂度: 'simple', 'medium', 'complex'
    processing_path: List[str]                           # 处理路径记录
    final_answer: str                                    # 最终答案


def analyze_query(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """
    分析用户查询，确定查询类型和复杂度

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    analysis_prompt = f"""分析以下用户查询，确定其类型和复杂度：

查询: {state['user_query']}

请以JSON格式返回分析结果：
{{
    "query_type": "factual|analytical|creative|unknown",
    "complexity_level": "simple|medium|complex",
    "reasoning": "分析理由"
}}

类型说明：
- factual: 事实性问题，需要查找具体信息
- analytical: 分析性问题，需要推理和计算
- creative: 创意性问题，需要生成新内容
- unknown: 无法确定类型

复杂度说明：
- simple: 简单直接的问题
- medium: 需要一定分析的问题
- complex: 复杂多步骤的问题"""

    response = llm.invoke([HumanMessage(content=analysis_prompt)])

    # 解析响应（简化版，实际应该用JSON解析）
    response_text = response.content.strip()

    # 简单解析（实际项目中应该用更健壮的解析）
    query_type = "unknown"
    complexity_level = "medium"

    if "factual" in response_text.lower():
        query_type = "factual"
    elif "analytical" in response_text.lower():
        query_type = "analytical"
    elif "creative" in response_text.lower():
        query_type = "creative"

    if "simple" in response_text.lower():
        complexity_level = "simple"
    elif "complex" in response_text.lower():
        complexity_level = "complex"

    return {
        "messages": [AIMessage(content=f"查询分析完成：类型={query_type}, 复杂度={complexity_level}")],
        "query_type": query_type,
        "complexity_level": complexity_level,
        "processing_path": ["analyze_query"]
    }


def handle_factual_query(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """处理事实性查询"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # 模拟知识库查询（实际项目中可以连接真实的知识库）
    knowledge_base = {
        "python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
        "ai": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。",
        "machine learning": "机器学习是人工智能的一个子集，使用算法从数据中学习模式。",
        "neural network": "神经网络是一种受生物神经系统启发的计算模型。",
    }

    query_lower = state['user_query'].lower()
    answer = "抱歉，我在知识库中没有找到相关信息。"

    # 简单关键词匹配
    for key, value in knowledge_base.items():
        if key in query_lower:
            answer = value
            break

    # 如果复杂度较高，进行更深入的处理
    if state['complexity_level'] == 'complex':
        # 使用LLM进行更深入的解释
        explain_prompt = f"""基于以下事实信息，提供更详细的解释：

事实: {answer}

查询: {state['user_query']}

请提供详细、准确的解释。"""

        detailed_response = llm.invoke([HumanMessage(content=explain_prompt)])
        answer = detailed_response.content

    return {
        "messages": [AIMessage(content=f"事实查询结果：{answer}")],
        "processing_path": state['processing_path'] + ["factual_handler"],
        "final_answer": answer
    }


def handle_analytical_query(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """处理分析性查询"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    analysis_prompt = f"""请对以下查询进行分析和推理：

查询: {state['user_query']}

请提供：
1. 问题分解
2. 相关因素分析
3. 推理过程
4. 结论

如果是{state['complexity_level']}复杂度的问题，请相应调整分析深度。"""

    response = llm.invoke([HumanMessage(content=analysis_prompt)])

    return {
        "messages": [AIMessage(content=f"分析结果：{response.content[:100]}...")],
        "processing_path": state['processing_path'] + ["analytical_handler"],
        "final_answer": response.content
    }


def handle_creative_query(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """处理创意性查询"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

    creative_prompt = f"""请基于以下查询生成创意内容：

查询: {state['user_query']}

要求：
- 创意新颖，有想象力
- 内容丰富，引人入胜
- 结构清晰，逻辑合理
- 根据{state['complexity_level']}复杂度调整内容长度和深度"""

    response = llm.invoke([HumanMessage(content=creative_prompt)])

    return {
        "messages": [AIMessage(content=f"创意内容：{response.content[:100]}...")],
        "processing_path": state['processing_path'] + ["creative_handler"],
        "final_answer": response.content
    }


def handle_unknown_query(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """处理未知类型查询"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    clarification_prompt = f"""用户查询不够清晰或类型不明，请尝试理解并提供帮助：

查询: {state['user_query']}

请：
1. 尝试理解用户的意图
2. 请求澄清如果需要
3. 提供相关的通用建议
4. 引导用户提供更多信息"""

    response = llm.invoke([HumanMessage(content=clarification_prompt)])

    return {
        "messages": [AIMessage(content=f"澄清建议：{response.content[:100]}...")],
        "processing_path": state['processing_path'] + ["unknown_handler"],
        "final_answer": response.content
    }


def route_by_query_type(state: ConditionalWorkflowState) -> str:
    """
    根据查询类型路由到相应的处理节点

    Args:
        state: 当前状态

    Returns:
        下一个节点名称
    """
    query_type = state.get('query_type', 'unknown')

    if query_type == 'factual':
        return 'factual_handler'
    elif query_type == 'analytical':
        return 'analytical_handler'
    elif query_type == 'creative':
        return 'creative_handler'
    else:
        return 'unknown_handler'


def format_final_response(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """格式化最终响应"""
    processing_path = ' -> '.join(state['processing_path'])
    final_answer = state['final_answer']

    formatted_response = f"""
处理路径: {processing_path}
查询类型: {state['query_type']}
复杂度: {state['complexity_level']}

答案:
{final_answer}
"""

    return {
        "messages": [AIMessage(content=f"最终响应已格式化")],
        "final_answer": formatted_response
    }


def create_conditional_workflow_graph():
    """
    创建条件分支工作流图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(ConditionalWorkflowState)

    # 添加节点
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("factual_handler", handle_factual_query)
    graph.add_node("analytical_handler", handle_analytical_query)
    graph.add_node("creative_handler", handle_creative_query)
    graph.add_node("unknown_handler", handle_unknown_query)
    graph.add_node("format_response", format_final_response)

    # 定义边
    graph.add_edge(START, "analyze_query")

    # 条件分支：根据查询类型路由
    graph.add_conditional_edges(
        "analyze_query",
        route_by_query_type,
        {
            "factual_handler": "factual_handler",
            "analytical_handler": "analytical_handler",
            "creative_handler": "creative_handler",
            "unknown_handler": "unknown_handler"
        }
    )

    # 所有处理节点都连接到格式化节点
    graph.add_edge("factual_handler", "format_response")
    graph.add_edge("analytical_handler", "format_response")
    graph.add_edge("creative_handler", "format_response")
    graph.add_edge("unknown_handler", "format_response")

    graph.add_edge("format_response", END)

    return graph.compile()


def demonstrate_conditional_flows():
    """
    演示条件分支工作流
    """
    print("🔀 条件分支工作流演示")
    print("系统会根据查询类型和复杂度自动选择处理路径")
    print("-" * 60)

    # 创建工作流
    workflow = create_conditional_workflow_graph()

    # 示例查询
    sample_queries = [
        "什么是Python编程语言？",  # factual, simple
        "人工智能将如何影响未来的就业市场？",  # analytical, complex
        "写一个关于未来城市的短故事",  # creative, medium
        "帮我分析一下这个数据",  # unknown/unclear
    ]

    print("示例查询：")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")

    while True:
        user_input = input("\n请选择查询编号 (1-4) 或输入自定义查询: ").strip()

        if user_input in ['1', '2', '3', '4']:
            query = sample_queries[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("🔀 再见！")
            break
        elif user_input:
            query = user_input
        else:
            print("❌ 请输入有效查询")
            continue

        print(f"\n🚀 处理查询: {query}")
        print("=" * 60)

        try:
            # 执行工作流
            result = workflow.invoke({
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "query_type": "",
                "complexity_level": "",
                "processing_path": [],
                "final_answer": ""
            })

            # 显示处理过程
            print("\n🔄 处理过程:")
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"🤖 {msg.content}")

            # 显示最终结果
            print("\n📋 最终结果:")
            print("-" * 40)
            print(result["final_answer"])

            print("\n✅ 处理完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 处理过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        demonstrate_conditional_flows()
    except KeyboardInterrupt:
        print("\n🔀 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



