#!/usr/bin/env python3
"""
循环和迭代处理示例

这个示例展示了如何在LangGraph中实现循环逻辑，
包括有限循环、无限循环保护、条件循环等。
"""

import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class IterativeWorkflowState(TypedDict):
    """迭代工作流状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史
    task: str                                            # 原始任务
    current_iteration: int                               # 当前迭代次数
    max_iterations: int                                  # 最大迭代次数
    iteration_results: Annotated[List[dict], operator.add]  # 每次迭代结果
    convergence_threshold: float                         # 收敛阈值
    previous_score: float                                # 上一次评分
    current_score: float                                 # 当前评分
    should_continue: bool                                # 是否继续迭代
    final_result: str                                    # 最终结果


def initialize_iteration(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """初始化迭代过程"""
    return {
        "messages": [AIMessage(content="开始迭代优化过程...")],
        "current_iteration": 0,
        "max_iterations": 5,  # 防止无限循环
        "convergence_threshold": 0.1,  # 收敛阈值
        "previous_score": 0.0,
        "current_score": 0.0,
        "should_continue": True
    }


def content_generation_step(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """内容生成步骤"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    iteration = state["current_iteration"]
    task = state["task"]

    # 根据迭代次数调整生成策略
    if iteration == 0:
        # 第一次生成：基础版本
        prompt = f"""请基于以下任务生成内容：

任务: {task}

要求：
- 提供基础版本的内容
- 结构清晰，逻辑合理
- 字数适中"""
    else:
        # 后续迭代：基于反馈改进
        previous_results = state.get("iteration_results", [])
        if previous_results:
            last_feedback = previous_results[-1].get("feedback", "")
            prompt = f"""请基于以下任务和改进建议重新生成内容：

任务: {task}

上一次反馈: {last_feedback}

要求：
- 参考反馈意见进行改进
- 保持核心内容不变
- 提升质量和准确性
- 这是第{iteration + 1}次迭代"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"第{iteration + 1}次生成完成")],
        "current_iteration": iteration + 1,
        "iteration_results": [{
            "iteration": iteration + 1,
            "content": response.content,
            "timestamp": "2024-01-01T00:00:00Z"  # 简化时间戳
        }]
    }


def quality_evaluation_step(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """质量评估步骤"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    current_content = ""
    iteration_results = state.get("iteration_results", [])
    if iteration_results:
        current_content = iteration_results[-1].get("content", "")

    task = state["task"]

    evaluation_prompt = f"""请评估以下内容的质量：

任务: {task}

内容:
{current_content}

请从以下维度进行评分（0-10分）：
1. 相关性：内容与任务的相关程度
2. 准确性：信息的准确程度
3. 完整性：内容的完整程度
4. 可读性：内容的清晰度和可读性
5. 创意性：内容的创新程度

请给出：
- 各维度评分
- 总体评分
- 改进建议

以JSON格式返回。"""

    response = llm.invoke([HumanMessage(content=evaluation_prompt)])

    # 解析评分（简化版）
    response_text = response.content
    current_score = 7.5  # 默认分数

    # 简单解析（实际项目中应该用JSON解析）
    if "8" in response_text or "9" in response_text or "10" in response_text:
        current_score = 8.5
    elif "6" in response_text or "7" in response_text:
        current_score = 7.0
    elif "4" in response_text or "5" in response_text:
        current_score = 6.0
    else:
        current_score = 5.5

    # 更新评分历史
    previous_score = state.get("previous_score", 0.0)

    return {
        "messages": [AIMessage(content=f"质量评估完成，评分: {current_score}")],
        "current_score": current_score,
        "previous_score": previous_score
    }


def improvement_suggestion_step(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """改进建议步骤"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    current_content = ""
    iteration_results = state.get("iteration_results", [])
    if iteration_results:
        current_content = iteration_results[-1].get("content", "")

    current_score = state.get("current_score", 0.0)
    task = state["task"]

    if current_score < 7.0:  # 需要改进
        improvement_prompt = f"""请分析以下内容并提供具体改进建议：

任务: {task}

当前内容:
{current_content}

当前评分: {current_score}

请提供：
1. 主要问题点
2. 具体改进建议
3. 优先改进项
4. 预期改进效果

改进建议要具体可操作。"""

        response = llm.invoke([HumanMessage(content=improvement_prompt)])

        return {
            "messages": [AIMessage(content="改进建议生成完成")],
            "iteration_results": [{
                "feedback": response.content
            }]
        }
    else:
        return {
            "messages": [AIMessage(content="内容质量已达标")],
            "iteration_results": [{
                "feedback": "内容质量良好，无需进一步改进"
            }]
        }


def check_convergence(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """检查收敛条件"""
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    current_score = state.get("current_score", 0.0)
    previous_score = state.get("previous_score", 0.0)
    convergence_threshold = state.get("convergence_threshold", 0.1)

    # 检查停止条件
    should_stop = False
    reason = ""

    # 条件1: 达到最大迭代次数
    if current_iteration >= max_iterations:
        should_stop = True
        reason = f"达到最大迭代次数 ({max_iterations})"

    # 条件2: 质量评分足够高
    elif current_score >= 8.5:
        should_stop = True
        reason = f"质量评分达标 ({current_score})"

    # 条件3: 收敛（评分变化很小）
    elif abs(current_score - previous_score) < convergence_threshold and current_iteration > 1:
        should_stop = True
        reason = f"评分收敛 (变化: {abs(current_score - previous_score):.2f})"

    # 条件4: 质量没有改善且已尝试多次
    elif current_iteration >= 3 and current_score <= previous_score:
        should_stop = True
        reason = "质量未改善，停止迭代"

    return {
        "messages": [AIMessage(content=f"收敛检查: {reason}")],
        "should_continue": not should_stop
    }


def finalize_result(state: IterativeWorkflowState) -> IterativeWorkflowState:
    """最终化结果"""
    iteration_results = state.get("iteration_results", [])
    final_content = ""

    # 选择最好的结果
    if iteration_results:
        # 简单选择最后一个结果（实际可以基于评分选择）
        final_content = iteration_results[-1].get("content", "")

    iteration_count = state.get("current_iteration", 0)
    final_score = state.get("current_score", 0.0)

    summary = f"""
迭代优化完成！

迭代次数: {iteration_count}
最终评分: {final_score}
总生成内容数: {len(iteration_results)}

最终内容:
{final_content}
"""

    return {
        "messages": [AIMessage(content="迭代优化过程完成")],
        "final_result": summary
    }


def should_iterate_again(state: IterativeWorkflowState) -> str:
    """
    决定是否继续迭代

    Args:
        state: 当前状态

    Returns:
        下一个节点
    """
    should_continue = state.get("should_continue", False)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    if should_continue and current_iteration < max_iterations:
        return "content_generation"
    else:
        return "finalize"


def create_iterative_workflow_graph():
    """
    创建迭代工作流图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(IterativeWorkflowState)

    # 添加节点
    graph.add_node("initialize", initialize_iteration)
    graph.add_node("content_generation", content_generation_step)
    graph.add_node("quality_evaluation", quality_evaluation_step)
    graph.add_node("improvement_suggestion", improvement_suggestion_step)
    graph.add_node("check_convergence", check_convergence)
    graph.add_node("finalize", finalize_result)

    # 定义边
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "content_generation")
    graph.add_edge("content_generation", "quality_evaluation")
    graph.add_edge("quality_evaluation", "improvement_suggestion")
    graph.add_edge("improvement_suggestion", "check_convergence")

    # 条件边：决定是否继续迭代
    graph.add_conditional_edges(
        "check_convergence",
        should_iterate_again,
        {
            "content_generation": "content_generation",
            "finalize": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()


def demonstrate_iterative_workflow():
    """
    演示迭代工作流
    """
    print("🔄 迭代优化工作流演示")
    print("系统会通过多次迭代不断改进内容质量")
    print("-" * 60)

    # 创建工作流
    workflow = create_iterative_workflow_graph()

    # 示例任务
    sample_tasks = [
        "写一篇关于人工智能的简介",
        "设计一个简单的项目计划",
        "创作一个短故事开头"
    ]

    print("示例任务：")
    for i, task in enumerate(sample_tasks, 1):
        print(f"{i}. {task}")

    while True:
        user_input = input("\n请选择任务编号 (1-3) 或输入自定义任务: ").strip()

        if user_input in ['1', '2', '3']:
            task = sample_tasks[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("🔄 再见！")
            break
        elif user_input:
            task = user_input
        else:
            print("❌ 请输入有效任务")
            continue

        print(f"\n🚀 开始迭代优化: {task}")
        print("=" * 60)

        try:
            # 执行工作流
            result = workflow.invoke({
                "messages": [HumanMessage(content=task)],
                "task": task,
                "current_iteration": 0,
                "iteration_results": [],
                "final_result": ""
            })

            # 显示迭代过程
            print("\n🔄 迭代过程:")
            for i, msg in enumerate(result["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"步骤 {i+1}: {msg.content}")

            # 显示最终结果
            print("\n📋 最终结果:")
            print("-" * 40)
            print(result["final_result"])

            print("\n✅ 迭代优化完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        demonstrate_iterative_workflow()
    except KeyboardInterrupt:
        print("\n🔄 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



