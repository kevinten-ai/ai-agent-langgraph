#!/usr/bin/env python3
"""
AI Agent LangGraph 多Agent协作系统演示

展示完整的多Agent工作流执行过程。
"""

import os
import asyncio
from dotenv import load_dotenv

from src import MultiAgentWorkflow

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 请在.env文件中设置OPENAI_API_KEY")
    exit(1)


async def demo_workflow():
    """演示工作流执行"""
    print("🤖 AI Agent LangGraph 多Agent协作系统演示")
    print("=" * 60)

    # 创建工作流实例
    workflow = MultiAgentWorkflow(enable_mcp_integration=True)

    # 示例任务
    sample_tasks = [
        "分析当前Python在数据科学领域的应用情况",
        "设计一个简单的用户管理系统架构",
        "研究人工智能在医疗诊断中的潜力"
    ]

    print("可用的示例任务：")
    for i, task in enumerate(sample_tasks, 1):
        print(f"{i}. {task}")

    while True:
        try:
            user_input = input("\n请选择任务编号 (1-3) 或输入自定义任务: ").strip()

            if user_input in ['1', '2', '3']:
                task = sample_tasks[int(user_input) - 1]
            elif user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            elif user_input:
                task = user_input
            else:
                print("❌ 请输入有效任务")
                continue

            print(f"\n🚀 开始执行任务: {task}")
            print("=" * 60)

            # 执行工作流
            result = await workflow.run_workflow(task)

            if result["success"]:
                print("✅ 任务执行成功！")
                print(f"📋 工作流ID: {result['workflow_id']}")
                print(f"💡 最终答案:\n{result['final_answer']}")

                # 显示执行摘要
                summary = result.get("execution_summary", {})
                if summary:
                    print("\n📊 执行摘要:")
                    print(f"  - 任务类型: {summary.get('task_type', 'N/A')}")
                    print(f"  - 执行状态: {summary.get('execution_status', 'N/A')}")
                    print(f"  - 审核分数: {summary.get('review_score', 'N/A')}")
                    print(f"  - 重试次数: {summary.get('retry_count', 0)}")
                    print(f"  - 工具调用: {summary.get('tool_calls', 0)}")
                    print(f"  - 执行时长: {summary.get('execution_time', 0):.2f}秒")
            else:
                print(f"❌ 执行失败: {result.get('error', '未知错误')}")

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 演示过程中发生错误: {str(e)}")


async def demo_status_tracking():
    """演示状态跟踪功能"""
    print("\n📊 工作流状态跟踪演示")
    print("-" * 40)

    workflow = MultiAgentWorkflow()

    # 启动一个长时间运行的任务
    task = "进行详细的市场分析报告，包括数据收集、趋势分析和建议"

    print(f"启动任务: {task}")

    # 异步执行任务
    import threading
    import time

    result_future = None

    def run_workflow():
        nonlocal result_future
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_future = loop.run_until_complete(workflow.run_workflow(task))

    workflow_thread = threading.Thread(target=run_workflow)
    workflow_thread.start()

    # 监控状态
    workflow_id = None
    while workflow_thread.is_alive():
        if result_future:
            # 获取工作流ID（简化实现）
            status = await workflow.get_workflow_status("temp_id")
            if status:
                print(f"当前状态: {status['status']}, 进度: {status['progress']:.1%}")
            time.sleep(2)
        else:
            print("正在初始化工作流...")
            time.sleep(1)

    workflow_thread.join()

    if result_future and result_future["success"]:
        print("✅ 任务完成！")
    else:
        print("❌ 任务失败")


def show_system_info():
    """显示系统信息"""
    print("\nℹ️  系统信息")
    print("-" * 40)

    workflow = MultiAgentWorkflow()

    # 显示MCP工具统计
    if hasattr(workflow, 'tool_registry'):
        registry_stats = workflow.tool_registry.get_statistics()
        print(f"📚 注册的MCP工具: {registry_stats['total_tools']}")
        print(f"🔧 工具类别: {registry_stats['categories']}")

    # 显示状态管理器统计
    if hasattr(workflow, 'state_manager'):
        state_stats = workflow.state_manager.get_statistics()
        print(f"📊 活跃状态: {state_stats['total_states']}")

    print(f"🤖 MCP集成: {'启用' if workflow.enable_mcp_integration else '禁用'}")


async def main():
    """主函数"""
    try:
        show_system_info()
        await demo_workflow()

        # 可选：演示状态跟踪
        show_status_demo = input("\n是否演示状态跟踪功能? (y/n): ").lower().strip()
        if show_status_demo == 'y':
            await demo_status_tracking()

    except Exception as e:
        print(f"❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())



