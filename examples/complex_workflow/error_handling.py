#!/usr/bin/env python3
"""
错误处理工作流示例

这个示例展示了如何在LangGraph中实现健壮的错误处理机制，
包括异常捕获、恢复策略、重试逻辑等。
"""

import os
import time
import random
from typing import TypedDict, Annotated, List, Optional
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class ErrorHandlingState(TypedDict):
    """错误处理工作流状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史
    task: str                                            # 原始任务
    current_attempt: int                                 # 当前尝试次数
    max_attempts: int                                    # 最大尝试次数
    errors: Annotated[List[dict], operator.add]         # 错误记录
    recovery_strategy: str                              # 恢复策略
    last_error: Optional[dict]                          # 最后一次错误
    success: bool                                       # 是否成功
    final_result: str                                   # 最终结果


def initialize_error_handling(state: ErrorHandlingState) -> ErrorHandlingState:
    """初始化错误处理"""
    return {
        "messages": [AIMessage(content="开始执行任务（带错误处理）...")],
        "current_attempt": 0,
        "max_attempts": 3,
        "errors": [],
        "recovery_strategy": "retry",
        "last_error": None,
        "success": False
    }


def risky_operation(state: ErrorHandlingState) -> ErrorHandlingState:
    """
    模拟可能失败的操作

    这个函数故意包含一些可能失败的逻辑来演示错误处理
    """
    current_attempt = state.get("current_attempt", 0) + 1

    print(f"🔄 执行尝试 #{current_attempt}")

    # 模拟不同的错误场景
    if current_attempt == 1:
        # 第一次尝试：网络错误
        error_type = "network_error"
        error_message = "网络连接超时"
    elif current_attempt == 2:
        # 第二次尝试：API限流
        error_type = "rate_limit"
        error_message = "API请求频率超限"
    elif current_attempt == 3:
        # 第三次尝试：内容过滤
        error_type = "content_filter"
        error_message = "内容被安全过滤器拦截"
    else:
        # 后续尝试：随机成功或失败
        if random.random() < 0.3:  # 30%成功率
            error_type = None
            error_message = None
        else:
            error_type = "unexpected_error"
            error_message = "未知错误"

    if error_type:
        # 记录错误
        error_record = {
            "attempt": current_attempt,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time(),
            "task": state["task"]
        }

        raise Exception(f"{error_type}: {error_message}")
    else:
        # 模拟成功执行任务
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        success_prompt = f"""请成功完成以下任务：

任务: {state['task']}

要求：
- 提供高质量的回答
- 结构清晰，内容完整
- 这是第{current_attempt}次尝试，之前失败了{current_attempt-1}次"""

        response = llm.invoke([HumanMessage(content=success_prompt)])

        return {
            "messages": [AIMessage(content=f"任务执行成功（尝试 #{current_attempt}）")],
            "current_attempt": current_attempt,
            "success": True,
            "final_result": response.content
        }


def error_handler(state: ErrorHandlingState) -> ErrorHandlingState:
    """错误处理器"""
    last_error = state.get("last_error", {})
    current_attempt = state.get("current_attempt", 0)
    max_attempts = state.get("max_attempts", 3)

    error_type = last_error.get("error_type", "unknown")
    error_message = last_error.get("error_message", "未知错误")

    print(f"❌ 检测到错误: {error_type} - {error_message}")

    # 根据错误类型选择恢复策略
    if error_type == "network_error":
        recovery_strategy = "retry_with_backoff"
        wait_time = min(2 ** current_attempt, 30)  # 指数退避，最多30秒
        print(f"⏳ 网络错误，使用退避重试策略，等待 {wait_time} 秒")
        time.sleep(wait_time)

    elif error_type == "rate_limit":
        recovery_strategy = "retry_later"
        wait_time = 10  # 固定等待时间
        print(f"⏳ API限流，等待 {wait_time} 秒后重试")
        time.sleep(wait_time)

    elif error_type == "content_filter":
        recovery_strategy = "modify_content"
        print("📝 内容被过滤，尝试修改内容后重试")

    elif error_type == "unexpected_error":
        recovery_strategy = "retry"
        print("🔄 未知错误，直接重试")

    else:
        recovery_strategy = "fail"
        print("💥 不可恢复错误")

    # 检查是否达到最大尝试次数
    if current_attempt >= max_attempts:
        recovery_strategy = "fail"
        print(f"🚫 已达到最大尝试次数 ({max_attempts})")

    return {
        "messages": [AIMessage(content=f"错误处理完成，选择策略: {recovery_strategy}")],
        "recovery_strategy": recovery_strategy
    }


def recovery_executor(state: ErrorHandlingState) -> ErrorHandlingState:
    """恢复执行器"""
    strategy = state.get("recovery_strategy", "retry")

    if strategy == "retry_with_backoff":
        # 退避重试已在error_handler中处理
        return {
            "messages": [AIMessage(content="退避重试完成，准备重新执行")],
        }

    elif strategy == "retry_later":
        # 限流等待已在error_handler中处理
        return {
            "messages": [AIMessage(content="限流等待完成，准备重新执行")],
        }

    elif strategy == "modify_content":
        # 修改内容策略
        task = state["task"]
        # 简化版：添加安全前缀
        modified_task = f"[安全模式] {task}"

        return {
            "messages": [AIMessage(content="内容已修改，准备重新执行")],
            "task": modified_task
        }

    elif strategy == "fail":
        # 失败处理
        error_summary = "任务执行失败，所有恢复策略都已尝试。\n\n错误记录:\n"
        for error in state.get("errors", []):
            error_summary += f"- 尝试 #{error['attempt']}: {error['error_type']} - {error['error_message']}\n"

        return {
            "messages": [AIMessage(content="所有恢复策略失败")],
            "success": False,
            "final_result": error_summary
        }

    else:
        # 默认重试
        return {
            "messages": [AIMessage(content="使用默认重试策略")],
        }


def should_retry(state: ErrorHandlingState) -> str:
    """
    决定是否重试

    Args:
        state: 当前状态

    Returns:
        下一个节点
    """
    strategy = state.get("recovery_strategy", "retry")
    current_attempt = state.get("current_attempt", 0)
    max_attempts = state.get("max_attempts", 3)

    if strategy == "fail" or current_attempt >= max_attempts:
        return "final_report"
    else:
        return "risky_operation"


def final_report_generator(state: ErrorHandlingState) -> ErrorHandlingState:
    """生成最终报告"""
    success = state.get("success", False)
    attempts = state.get("current_attempt", 0)
    errors = state.get("errors", [])

    if success:
        result = state.get("final_result", "")
        report = f"""✅ 任务执行成功！

执行统计:
- 总尝试次数: {attempts}
- 失败次数: {len(errors)}
- 最终状态: 成功

结果:
{result}
"""
    else:
        report = f"""❌ 任务执行失败

执行统计:
- 总尝试次数: {attempts}
- 失败次数: {len(errors)}
- 最终状态: 失败

失败原因:
{state.get('final_result', '未知错误')}

建议:
1. 检查网络连接
2. 验证API密钥
3. 调整任务内容
4. 联系技术支持
"""

    return {
        "messages": [AIMessage(content="最终报告已生成")],
        "final_result": report
    }


def create_error_handling_graph():
    """
    创建错误处理工作流图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(ErrorHandlingState)

    # 添加节点
    graph.add_node("initialize", initialize_error_handling)
    graph.add_node("risky_operation", risky_operation)
    graph.add_node("error_handler", error_handler)
    graph.add_node("recovery_executor", recovery_executor)
    graph.add_node("final_report", final_report_generator)

    # 定义边
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "risky_operation")

    # 错误处理边（通过异常处理）
    # 注意：实际实现中需要使用LangGraph的异常处理机制

    graph.add_edge("error_handler", "recovery_executor")

    # 条件边：决定是否重试
    graph.add_conditional_edges(
        "recovery_executor",
        should_retry,
        {
            "risky_operation": "risky_operation",
            "final_report": "final_report"
        }
    )

    graph.add_edge("final_report", END)

    return graph.compile()


def demonstrate_error_handling():
    """
    演示错误处理工作流

    注意：这个演示使用了简化的错误处理逻辑。
    在实际的LangGraph中，应该使用内置的异常处理机制。
    """
    print("🛡️ 错误处理工作流演示")
    print("系统会自动处理各种错误并尝试恢复")
    print("-" * 60)

    # 示例任务（故意设计一些可能触发错误的任务）
    sample_tasks = [
        "写一篇关于人工智能的文章",
        "分析当前科技趋势",
        "创作一个短故事"
    ]

    print("示例任务：")
    for i, task in enumerate(sample_tasks, 1):
        print(f"{i}. {task}")

    while True:
        user_input = input("\n请选择任务编号 (1-3) 或输入自定义任务: ").strip()

        if user_input in ['1', '2', '3']:
            task = sample_tasks[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("🛡️ 再见！")
            break
        elif user_input:
            task = user_input
        else:
            print("❌ 请输入有效任务")
            continue

        print(f"\n🚀 开始执行任务（带错误处理）: {task}")
        print("=" * 60)

        # 模拟错误处理流程（简化版）
        try:
            # 这里我们直接调用risky_operation来演示错误处理
            # 实际应该使用完整的图执行

            state = {
                "messages": [HumanMessage(content=task)],
                "task": task,
                "current_attempt": 0,
                "max_attempts": 3,
                "errors": [],
                "success": False,
                "final_result": ""
            }

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    print(f"\n🔄 尝试 #{attempt + 1}")
                    result = risky_operation({
                        **state,
                        "current_attempt": attempt
                    })

                    if result.get("success", False):
                        print("✅ 执行成功！")
                        final_result = result["final_result"]
                        break

                except Exception as e:
                    print(f"❌ 尝试 #{attempt + 1} 失败: {str(e)}")

                    # 记录错误
                    error_record = {
                        "attempt": attempt + 1,
                        "error_type": str(e).split(":")[0] if ":" in str(e) else "unknown",
                        "error_message": str(e),
                        "timestamp": time.time(),
                        "task": task
                    }

                    state["errors"].append(error_record)

                    if attempt == max_attempts - 1:
                        # 最后一次尝试失败
                        final_result = "任务执行失败，所有重试都已尝试。"
                        for error in state["errors"]:
                            final_result += f"\n- 尝试 #{error['attempt']}: {error['error_message']}"
                        break

                    # 等待后重试（模拟退避策略）
                    wait_time = min(2 ** attempt, 10)
                    print(f"⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

            # 显示结果
            print("\n📋 执行结果:")
            print("-" * 40)
            print(final_result)

            print("\n✅ 错误处理演示完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 演示过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        demonstrate_error_handling()
    except KeyboardInterrupt:
        print("\n🛡️ 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



