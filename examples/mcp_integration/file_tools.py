#!/usr/bin/env python3
"""
MCP文件系统工具集成示例

这个示例展示了如何将MCP协议的文件系统工具集成到LangGraph Agent中，
实现文件读写、目录操作等功能。
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class MCPFileState(TypedDict):
    """MCP文件工具状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史
    user_query: str                                      # 用户查询
    file_operations: Annotated[List[Dict[str, Any]], operator.add]  # 文件操作记录
    current_file: str                                    # 当前操作的文件
    workspace_root: str                                  # 工作空间根目录
    final_answer: str                                    # 最终答案


# 模拟MCP文件系统工具
@tool
def mcp_read_file(path: str, encoding: str = "utf-8") -> str:
    """
    MCP文件读取工具 - 读取文件内容

    Args:
        path: 文件路径
        encoding: 文件编码

    Returns:
        文件内容或错误信息
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(path):
            return f"错误: 文件不存在 '{path}'"

        # 检查是否是文件
        if not os.path.isfile(path):
            return f"错误: 路径不是文件 '{path}'"

        # 读取文件内容
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()

        return f"文件内容 ({path}):\n{content}"

    except UnicodeDecodeError:
        return f"错误: 无法以 {encoding} 编码读取文件 '{path}'"
    except PermissionError:
        return f"错误: 没有权限读取文件 '{path}'"
    except Exception as e:
        return f"错误: 读取文件时发生异常 '{path}': {str(e)}"


@tool
def mcp_list_directory(path: str) -> str:
    """
    MCP目录列出工具 - 列出目录内容

    Args:
        path: 目录路径

    Returns:
        目录内容列表
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(path):
            return f"错误: 目录不存在 '{path}'"

        # 检查是否是目录
        if not os.path.isdir(path):
            return f"错误: 路径不是目录 '{path}'"

        # 列出目录内容
        items = os.listdir(path)
        files = []
        directories = []

        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                files.append(f"文件: {item} ({size} bytes)")
            elif os.path.isdir(full_path):
                directories.append(f"目录: {item}/")

        result = f"目录内容 ({path}):\n"
        if directories:
            result += "\n目录:\n" + "\n".join(f"  {d}" for d in directories)
        if files:
            result += "\n文件:\n" + "\n".join(f"  {f}" for f in files)

        return result

    except PermissionError:
        return f"错误: 没有权限访问目录 '{path}'"
    except Exception as e:
        return f"错误: 列出目录时发生异常 '{path}': {str(e)}"


@tool
def mcp_write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    MCP文件写入工具 - 写入文件内容

    Args:
        path: 文件路径
        content: 要写入的内容
        encoding: 文件编码

    Returns:
        操作结果
    """
    try:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 写入文件
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

        return f"成功: 文件已写入 '{path}' ({len(content)} 字符)"

    except PermissionError:
        return f"错误: 没有权限写入文件 '{path}'"
    except Exception as e:
        return f"错误: 写入文件时发生异常 '{path}': {str(e)}"


@tool
def mcp_file_search(directory: str, pattern: str) -> str:
    """
    MCP文件搜索工具 - 在目录中搜索文件

    Args:
        directory: 搜索目录
        pattern: 搜索模式（文件名模式）

    Returns:
        搜索结果
    """
    try:
        import glob

        # 检查目录是否存在
        if not os.path.exists(directory):
            return f"错误: 目录不存在 '{directory}'"

        # 搜索文件
        search_path = os.path.join(directory, f"**/{pattern}")
        matches = glob.glob(search_path, recursive=True)

        if not matches:
            return f"在 '{directory}' 中没有找到匹配 '{pattern}' 的文件"

        result = f"搜索结果 (目录: {directory}, 模式: {pattern}):\n"
        for match in matches[:20]:  # 限制结果数量
            if os.path.isfile(match):
                size = os.path.getsize(match)
                result += f"  {match} ({size} bytes)\n"

        if len(matches) > 20:
            result += f"  ... 还有 {len(matches) - 20} 个结果\n"

        return result

    except Exception as e:
        return f"错误: 搜索文件时发生异常: {str(e)}"


def initialize_workspace(state: MCPFileState) -> MCPFileState:
    """初始化工作空间"""
    workspace_root = os.getcwd()  # 使用当前目录作为工作空间

    return {
        "messages": [AIMessage(content=f"工作空间已初始化: {workspace_root}")],
        "workspace_root": workspace_root,
        "file_operations": []
    }


def analyze_file_request(state: MCPFileState) -> MCPFileState:
    """分析文件操作请求"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    analysis_prompt = f"""分析用户的文件操作请求，并确定需要使用的MCP工具：

用户请求: {state['user_query']}

请确定：
1. 请求类型：read（读取）、write（写入）、list（列出）、search（搜索）
2. 目标路径：具体文件或目录路径
3. 其他参数：编码、搜索模式等

请以JSON格式返回分析结果：
{{
    "operation_type": "read|write|list|search",
    "target_path": "文件或目录路径",
    "parameters": {{
        "encoding": "utf-8",
        "pattern": "搜索模式",
        "content": "写入内容"
    }}
}}"""

    response = llm.invoke([HumanMessage(content=analysis_prompt)])

    # 解析响应（简化版）
    response_text = response.content.strip()

    # 默认分析结果
    operation_type = "read"
    target_path = state['user_query'].split()[-1] if len(state['user_query'].split()) > 1 else "."

    if "list" in response_text.lower() or "列出" in state['user_query']:
        operation_type = "list"
    elif "write" in response_text.lower() or "写入" in state['user_query'] or "创建" in state['user_query']:
        operation_type = "write"
    elif "search" in response_text.lower() or "搜索" in state['user_query'] or "查找" in state['user_query']:
        operation_type = "search"

    return {
        "messages": [AIMessage(content=f"请求分析完成: {operation_type} 操作")],
        "current_file": target_path
    }


def execute_file_operation(state: MCPFileState) -> MCPFileState:
    """执行文件操作"""
    operation_type = "read"  # 简化版，默认为读取
    target_path = state.get('current_file', '.')

    # 根据操作类型调用相应的MCP工具
    if operation_type == "read":
        result = mcp_read_file.invoke({"path": target_path})
    elif operation_type == "list":
        result = mcp_list_directory.invoke({"path": target_path})
    elif operation_type == "write":
        # 模拟写入内容
        content = f"# Generated by MCP Agent\n\nContent for {target_path}"
        result = mcp_write_file.invoke({
            "path": target_path,
            "content": content
        })
    elif operation_type == "search":
        pattern = "*.py"  # 默认搜索Python文件
        result = mcp_file_search.invoke({
            "directory": state['workspace_root'],
            "pattern": pattern
        })
    else:
        result = "错误: 不支持的操作类型"

    # 记录操作
    operation_record = {
        "type": operation_type,
        "path": target_path,
        "result": result,
        "timestamp": "2024-01-01T00:00:00Z"
    }

    return {
        "messages": [AIMessage(content=f"文件操作完成: {operation_type}")],
        "file_operations": [operation_record]
    }


def generate_response(state: MCPFileState) -> MCPFileState:
    """生成最终响应"""
    operations = state.get('file_operations', [])
    user_query = state['user_query']

    response = f"文件操作请求处理完成！\n\n原始请求: {user_query}\n\n"

    if operations:
        response += "执行的操作:\n"
        for op in operations:
            response += f"- {op['type']} 操作: {op['path']}\n"
            response += f"  结果: {op['result'][:200]}...\n\n"

    response += "\nMCP文件系统工具支持以下操作:\n"
    response += "- 读取文件内容 (read_file)\n"
    response += "- 列出目录内容 (list_directory)\n"
    response += "- 写入文件内容 (write_file)\n"
    response += "- 搜索文件 (file_search)\n"

    return {
        "messages": [AIMessage(content="响应生成完成")],
        "final_answer": response
    }


def create_mcp_file_graph():
    """
    创建MCP文件工具集成图

    Returns:
        编译后的状态图
    """
    graph = StateGraph(MCPFileState)

    # 添加节点
    graph.add_node("initialize_workspace", initialize_workspace)
    graph.add_node("analyze_request", analyze_file_request)
    graph.add_node("execute_operation", execute_file_operation)
    graph.add_node("generate_response", generate_response)

    # 定义边
    graph.add_edge(START, "initialize_workspace")
    graph.add_edge("initialize_workspace", "analyze_request")
    graph.add_edge("analyze_request", "execute_operation")
    graph.add_edge("execute_operation", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


def demonstrate_mcp_file_tools():
    """
    演示MCP文件工具集成
    """
    print("📁 MCP文件系统工具集成演示")
    print("Agent可以执行文件读写、目录操作等功能")
    print("-" * 60)

    # 创建MCP文件Agent
    file_agent = create_mcp_file_graph()

    # 示例请求
    sample_requests = [
        "读取README.md文件的内容",
        "列出当前目录的文件",
        "搜索所有的Python文件",
        "创建一个测试文件"
    ]

    print("示例请求：")
    for i, request in enumerate(sample_requests, 1):
        print(f"{i}. {request}")

    while True:
        user_input = input("\n请选择请求编号 (1-4) 或输入自定义请求: ").strip()

        if user_input in ['1', '2', '3', '4']:
            request = sample_requests[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("📁 再见！")
            break
        elif user_input:
            request = user_input
        else:
            print("❌ 请输入有效请求")
            continue

        print(f"\n🚀 执行文件操作: {request}")
        print("=" * 60)

        try:
            # 执行文件操作
            result = file_agent.invoke({
                "messages": [HumanMessage(content=request)],
                "user_query": request,
                "file_operations": [],
                "final_answer": ""
            })

            # 显示结果
            print("\n📋 执行结果:")
            print("-" * 40)
            print(result["final_answer"])

            # 显示操作记录
            operations = result.get("file_operations", [])
            if operations:
                print("\n🔧 操作详情:")
                for op in operations:
                    print(f"类型: {op['type']}")
                    print(f"路径: {op['path']}")
                    print(f"结果: {op['result'][:100]}...")
                    print()

            print("\n✅ 文件操作完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        demonstrate_mcp_file_tools()
    except KeyboardInterrupt:
        print("\n📁 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



