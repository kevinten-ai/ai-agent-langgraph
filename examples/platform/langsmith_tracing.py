#!/usr/bin/env python3
"""
LangSmith Tracing 示例

演示如何基于 src/workflow/orchestrator.py 中的 get_graph()，
在运行图时自动产生 LangSmith trace。

运行方式:
    python examples/platform/langsmith_tracing.py

说明:
- 设置 LANGCHAIN_TRACING_V2=true 即可自动追踪 LangGraph 执行链路。
- 若没有设置有效的 LANGSMITH_API_KEY，程序会正常执行，但不会产生远端 trace。
- 本示例使用了"模拟模式"，不依赖真实的 OpenAI API Key 也能跑通整个工作流。
"""

import os
import sys
import asyncio

# 将项目根目录加入 PYTHONPATH，以便导入 src 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ============================================================
# 1. 配置 LangSmith 环境变量（在导入 LangChain/LangGraph 相关模块之前设置最佳）
# ============================================================
# 开启 LangSmith Tracing V2
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 如果有 LangSmith API Key，取消下面一行的注释并填入真实值
# os.environ["LANGSMITH_API_KEY"] = "ls-your-api-key-here"

# 可选：指定项目名，方便在 LangSmith UI 中归类查看
os.environ["LANGCHAIN_PROJECT"] = "ai-agent-langgraph-demo"

# ============================================================
# 2. 导入项目组件
# ============================================================
from src.workflow.orchestrator import get_graph  # noqa: E402


async def main():
    """主函数：运行一次完整的图调用，观察 LangSmith 自动追踪效果。"""

    print("=" * 60)
    print("LangSmith Tracing 演示")
    print("=" * 60)

    # 检查当前追踪配置
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "")
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    print(f"\n[配置信息]")
    print(f"  LANGCHAIN_TRACING_V2 = {tracing_v2}")
    print(f"  LANGSMITH_API_KEY    = {'已设置' if api_key else '未设置'}")
    print(f"  LANGCHAIN_PROJECT    = {project}")

    if not api_key:
        print("\n[提示] 未检测到 LANGSMITH_API_KEY，程序将正常运行，但不会上报远端 trace。")

    # 获取编译好的图实例
    app = get_graph()
    print(f"\n[图信息] 节点列表: {list(app.nodes.keys())}")

    # ============================================================
    # 3. 构造输入并调用图
    # ============================================================
    user_input = "帮我简单介绍一下 LangGraph 的核心概念"
    print(f"\n[运行] 用户输入: {user_input}")

    # orchestrator 的输入是 TypedDict(AgentState)，核心字段至少包含 user_input
    # 如果环境中没有 OPENAI_API_KEY，MultiAgentWorkflow 会走模拟逻辑（内部 catch 使用 mock）
    # 但 orchestrator 本身不会自动 mock，所以我们尝试执行，若失败则打印提示。
    try:
        # ainvoke 传入初始状态
        result = await app.ainvoke({"user_input": user_input})
    except Exception as e:
        print(f"\n[异常] 图执行失败: {e}")
        print("[提示] 若失败原因是缺少 OPENAI_API_KEY，可设置环境变量后再试；")
        print("      或者检查一下 src/workflow/orchestrator.py 是否已处理无 Key 的降级逻辑。")
        return

    # ============================================================
    # 4. 输出结果
    # ============================================================
    print("\n[执行结果]")
    if isinstance(result, dict):
        final_answer = result.get("final_answer")
        workflow_status = result.get("workflow_status")
        error_messages = result.get("error_messages", [])

        print(f"  workflow_status : {workflow_status}")
        print(f"  final_answer    : {final_answer or '(空)'}")
        if error_messages:
            print(f"  error_messages  : {error_messages}")
    else:
        # result 可能是 AgentState (TypedDict/Pydantic)
        print(f"  返回类型: {type(result)}")
        print(f"  结果摘要: {str(result)[:500]}")

    print("\n" + "=" * 60)
    if api_key:
        print("✅ 演示完成。请前往 LangSmith Web 界面查看本次 Trace 详情。")
        print(f"   项目地址: https://smith.langchain.com/projects/{project}")
    else:
        print("✅ 演示完成（本地模式，未上报 LangSmith）。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
