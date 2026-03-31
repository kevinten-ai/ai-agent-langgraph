#!/usr/bin/env python3
"""
Subgraph 子图组合示例

演示:
1. 共享 Schema — 子图直接作为父图节点
2. 不同 Schema — 函数包装做 State 转换
3. 子图独立状态空间，不污染父图

关键 API:
- sub_graph = sub_builder.compile()          → 编译子图
- parent.add_node("sub", sub_graph)          → 共享 Schema 时直接嵌入
- parent.add_node("sub", wrapper_fn)         → 不同 Schema 时用函数包装
- subgraphs=True (stream 参数)               → 流式输出包含子图事件

LangGraph 概念: Subgraph, State Mapping, Graph Composition
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END


# ============================================================
# 演示 1: 共享 Schema — 子图直接嵌入
# ============================================================

class SharedState(TypedDict):
    """父图和子图共享的状态"""
    data: str
    steps: Annotated[List[str], lambda a, b: a + b]  # 追加归约器


def sub_step_a(state: SharedState) -> dict:
    """子图步骤 A"""
    return {
        "data": state["data"] + " → [子图A处理]",
        "steps": ["sub_step_a"]
    }


def sub_step_b(state: SharedState) -> dict:
    """子图步骤 B"""
    return {
        "data": state["data"] + " → [子图B处理]",
        "steps": ["sub_step_b"]
    }


def pre_process(state: SharedState) -> dict:
    """父图: 预处理"""
    return {
        "data": "[预处理] " + state["data"],
        "steps": ["pre_process"]
    }


def post_process(state: SharedState) -> dict:
    """父图: 后处理"""
    return {
        "data": state["data"] + " → [后处理完成]",
        "steps": ["post_process"]
    }


def demo_shared_schema():
    """演示共享 Schema 的子图"""
    print("=" * 60)
    print("演示 1: 共享 Schema — 子图直接作为节点")
    print("=" * 60)

    # 构建子图
    sub_builder = StateGraph(SharedState)
    sub_builder.add_node("step_a", sub_step_a)
    sub_builder.add_node("step_b", sub_step_b)
    sub_builder.add_edge(START, "step_a")
    sub_builder.add_edge("step_a", "step_b")
    sub_builder.add_edge("step_b", END)
    sub_graph = sub_builder.compile()

    # 构建父图 — 子图直接作为节点
    parent_builder = StateGraph(SharedState)
    parent_builder.add_node("pre", pre_process)
    parent_builder.add_node("sub", sub_graph)  # 关键: 子图作为节点
    parent_builder.add_node("post", post_process)
    parent_builder.add_edge(START, "pre")
    parent_builder.add_edge("pre", "sub")
    parent_builder.add_edge("sub", "post")
    parent_builder.add_edge("post", END)
    parent_app = parent_builder.compile()

    # 执行
    result = parent_app.invoke({"data": "原始数据", "steps": []})
    print(f"  数据流: {result['data']}")
    print(f"  执行步骤: {' → '.join(result['steps'])}")
    print()


# ============================================================
# 演示 2: 不同 Schema — 函数包装转换
# ============================================================

class ParentState(TypedDict):
    """父图状态"""
    user_query: str
    final_answer: str
    log: Annotated[List[str], lambda a, b: a + b]


class ChildState(TypedDict):
    """子图状态 — 与父图不同"""
    input_text: str
    processed_text: str
    word_count: int


def child_analyze(state: ChildState) -> dict:
    """子图: 文本分析"""
    text = state["input_text"]
    word_count = len(text)
    return {
        "processed_text": f"分析结果: '{text}' 共 {word_count} 字符",
        "word_count": word_count
    }


def child_enhance(state: ChildState) -> dict:
    """子图: 文本增强"""
    return {
        "processed_text": state["processed_text"] + " | 已增强优化"
    }


def create_child_graph():
    """构建子图"""
    builder = StateGraph(ChildState)
    builder.add_node("analyze", child_analyze)
    builder.add_node("enhance", child_enhance)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "enhance")
    builder.add_edge("enhance", END)
    return builder.compile()


def parent_init(state: ParentState) -> dict:
    """父图: 初始化"""
    return {"log": ["parent_init"]}


def child_wrapper(state: ParentState) -> dict:
    """
    包装函数: 负责 ParentState ↔ ChildState 转换

    这是不同 Schema 子图的核心模式:
    1. 从 ParentState 提取数据 → 构造 ChildState
    2. 调用子图
    3. 从子图结果提取数据 → 构造 ParentState 更新
    """
    child_app = create_child_graph()

    # ParentState → ChildState
    child_input = {
        "input_text": state["user_query"],
        "processed_text": "",
        "word_count": 0
    }

    # 调用子图
    child_result = child_app.invoke(child_input)

    # ChildState → ParentState
    return {
        "final_answer": child_result["processed_text"],
        "log": ["child_wrapper (子图执行完毕)"]
    }


def parent_finalize(state: ParentState) -> dict:
    """父图: 最终化"""
    return {"log": ["parent_finalize"]}


def demo_different_schema():
    """演示不同 Schema 的子图"""
    print("=" * 60)
    print("演示 2: 不同 Schema — 函数包装做 State 转换")
    print("=" * 60)

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("init", parent_init)
    parent_builder.add_node("child", child_wrapper)  # 包装函数作为节点
    parent_builder.add_node("finalize", parent_finalize)
    parent_builder.add_edge(START, "init")
    parent_builder.add_edge("init", "child")
    parent_builder.add_edge("child", "finalize")
    parent_builder.add_edge("finalize", END)
    parent_app = parent_builder.compile()

    # 执行
    result = parent_app.invoke({
        "user_query": "LangGraph 子图组合非常强大",
        "final_answer": "",
        "log": []
    })

    print(f"  输入: {result['user_query']}")
    print(f"  子图处理结果: {result['final_answer']}")
    print(f"  执行日志: {' → '.join(result['log'])}")
    print()


# ============================================================
# 演示 3: 流式输出子图事件
# ============================================================
def demo_subgraph_streaming():
    """演示子图的流式输出"""
    print("=" * 60)
    print("演示 3: 流式输出子图内部事件")
    print("=" * 60)

    # 复用演示 1 的图
    sub_builder = StateGraph(SharedState)
    sub_builder.add_node("step_a", sub_step_a)
    sub_builder.add_node("step_b", sub_step_b)
    sub_builder.add_edge(START, "step_a")
    sub_builder.add_edge("step_a", "step_b")
    sub_builder.add_edge("step_b", END)
    sub_graph = sub_builder.compile()

    parent_builder = StateGraph(SharedState)
    parent_builder.add_node("pre", pre_process)
    parent_builder.add_node("sub", sub_graph)
    parent_builder.add_node("post", post_process)
    parent_builder.add_edge(START, "pre")
    parent_builder.add_edge("pre", "sub")
    parent_builder.add_edge("sub", "post")
    parent_builder.add_edge("post", END)
    parent_app = parent_builder.compile()

    # 流式输出 — subgraphs=True 可以看到子图内部事件
    print("  流式事件:")
    for event in parent_app.stream(
        {"data": "测试数据", "steps": []},
        stream_mode="updates",
        subgraphs=True  # 关键: 包含子图事件
    ):
        # event 格式: (namespace_tuple, {node: update})
        if isinstance(event, tuple) and len(event) == 2:
            namespace, update_dict = event
            ns_str = " > ".join(namespace) if namespace else "root"
            for node_name in update_dict:
                print(f"    [{ns_str}] {node_name}")
        else:
            for node_name in event:
                print(f"    [root] {node_name}")
    print()


# ============================================================
# 主函数
# ============================================================
def main():
    print("🧩 Subgraph 子图组合演示\n")
    demo_shared_schema()
    demo_different_schema()
    demo_subgraph_streaming()
    print("✅ 所有演示完成")


if __name__ == "__main__":
    main()
