# FAQ & 踩坑指南

> 自学 LangGraph 最常遇到的问题和解决方案。每个坑都标注了**症状、原因、修复**。

---

## 一、图构建常见错误

### 1. "Found edge starting at unknown node"

**症状：** 编译时报错，提示某个边的起点节点不存在。

**原因：** `add_edge` 或 `add_conditional_edges` 中的节点名拼写错误，或忘记 `add_node`。

```python
# ❌ 错误：节点名拼写不一致
graph.add_node("analyser", fn)         # 注意拼写
graph.add_edge("analyzer", END)        # 拼写不同！

# ✅ 修复：保持一致
graph.add_node("analyzer", fn)
graph.add_edge("analyzer", END)
```

### 2. "Conditional edge function must return a string"

**症状：** 条件路由函数返回了非字符串值。

**原因：** 路由函数必须返回**节点名称字符串**（或 END）。

```python
# ❌ 错误：返回了 bool
def route(state):
    return state["ready"]  # 返回 True/False

# ✅ 修复：返回节点名
def route(state):
    return "execute" if state["ready"] else "wait"
```

### 3. 条件边映射表不完整

**症状：** 运行时报 `ValueError`，路由函数返回的值不在映射表中。

```python
# ❌ 错误：映射表缺少 "unknown" 选项
graph.add_conditional_edges("router", route_fn, {
    "math": "math_node",
    "write": "write_node",
    # route_fn 可能返回 "unknown"，但这里没有处理
})

# ✅ 修复：覆盖所有可能的返回值
graph.add_conditional_edges("router", route_fn, {
    "math": "math_node",
    "write": "write_node",
    "unknown": "fallback_node",  # 兜底
})
```

---

## 二、State 相关问题

### 4. 状态被覆盖而不是追加

**症状：** 多个节点往 `messages` 写入，但只保留了最后一个节点的消息。

**原因：** 忘记给列表字段加**归约器 (Reducer)**。

```python
# ❌ 错误：没有归约器，后写入的覆盖先写入的
class State(TypedDict):
    messages: list           # 覆盖语义！

# ✅ 修复：用 Annotated 加归约器
from langgraph.graph import add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]   # 追加语义
```

**速记规则：**
- 普通字段（`str`, `int`, `bool`）→ 不需要归约器，覆盖就对了
- 列表字段（`list`）→ 几乎总是需要 `operator.add` 或 `add_messages`
- 消息字段 → 用 `add_messages`（智能去重 + ID 追踪）

### 5. Pydantic State 忘记 `model_dump()`

**症状：** 用 Pydantic BaseModel 做 State 时，序列化报错。

```python
# ❌ 错误：直接传 Pydantic 对象
return {"task": task_object}  # task_object 是 Pydantic 实例

# ✅ 修复：在需要序列化时 dump
return {"task": task_object.model_dump()}
```

### 6. 节点返回了完整 State 而不是增量更新

**症状：** 某些字段被意外重置。

**原因：** 节点应该只返回**需要更新的字段**，而不是整个 State。

```python
# ❌ 错误：返回了所有字段，未改动的也会被覆盖
def my_node(state):
    return {
        "messages": state["messages"] + [new_msg],
        "result": "done",
        "counter": state["counter"],   # 不需要返回未改动的字段
        "data": state["data"],         # 不需要
    }

# ✅ 修复：只返回需要更新的
def my_node(state):
    return {
        "messages": [new_msg],   # 归约器会自动追加
        "result": "done"         # 覆盖旧值
    }
```

---

## 三、Checkpoint 问题

### 7. interrupt() 不生效 / 报错

**症状：** 调用 `interrupt()` 时报 `RuntimeError` 或直接跳过。

**原因：** 没有配置 Checkpointer。`interrupt()` 必须有 Checkpointer 来保存暂停时的状态。

```python
# ❌ 错误：没有 Checkpointer
app = graph.compile()  # interrupt() 无法工作

# ✅ 修复：加上 Checkpointer
from langgraph.checkpoint.memory import MemorySaver
app = graph.compile(checkpointer=MemorySaver())
```

### 8. 忘记传 thread_id

**症状：** 每次调用都是全新状态，没有记忆。

```python
# ❌ 错误：没有 config
result = app.invoke(input)  # 每次都是新会话

# ✅ 修复：传入 thread_id
config = {"configurable": {"thread_id": "my-session"}}
result = app.invoke(input, config=config)
```

### 9. MemorySaver 重启后数据丢失

**症状：** 程序重启后，之前的对话记忆消失。

**原因：** `MemorySaver` 是**内存级**的，进程退出即丢失。

```python
# 开发环境 OK
checkpointer = MemorySaver()

# 生产环境 → 换 PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

---

## 四、Streaming 问题

### 10. stream 看不到逐 token 输出

**症状：** `stream_mode="updates"` 或 `"values"` 只在节点**结束**后才输出。

**原因：** 这两种模式输出的是节点级别的，不是 token 级别的。

```python
# ❌ 这些模式看不到逐 token
app.stream(input, stream_mode="updates")   # 节点级
app.stream(input, stream_mode="values")    # 节点级

# ✅ 逐 token 用 messages 或 astream_events
app.stream(input, stream_mode="messages")  # 逐 token

# 或者用 astream_events 获取最细粒度
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### 11. 流式输出中看不到子图事件

**症状：** 子图内部的节点更新没有出现在流式输出中。

```python
# ❌ 错误：没有开启 subgraphs
for event in app.stream(input, stream_mode="updates"):
    ...  # 看不到子图内部

# ✅ 修复：加 subgraphs=True
for event in app.stream(input, stream_mode="updates", subgraphs=True):
    ...  # 可以看到子图内部事件
```

---

## 五、多 Agent 问题

### 12. 无限循环

**症状：** Agent 之间互相交接，永远不结束。

**原因：** 循环没有终止条件。

```python
# ❌ 错误：A → B → A → B → ...
def agent_a(state):
    return Command(goto="agent_b")
def agent_b(state):
    return Command(goto="agent_a")

# ✅ 修复：加计数器或条件判断
def agent_a(state):
    if state["handoff_count"] >= 3:
        return Command(goto=END)          # 强制结束
    return Command(
        update={"handoff_count": state["handoff_count"] + 1},
        goto="agent_b"
    )
```

**防护措施：**
- `max_iterations` / `handoff_count` 计数器
- LangGraph 默认递归限制 1000 步（会自动报错）
- `RemainingSteps` 可以在接近限制时优雅降级

### 13. Command vs add_conditional_edges 选择困难

| 场景 | 推荐 |
|------|------|
| 路由逻辑简单、固定 | `add_conditional_edges` |
| 节点内部动态决定去哪 | `Command(goto=...)` |
| 需要同时更新状态和路由 | `Command(update={...}, goto=...)` |
| Agent 间 handoff | `Command(goto=...)` |

---

## 六、环境与依赖问题

### 14. API Key 报错

```bash
# 症状: openai.AuthenticationError
# 检查:
echo $OPENAI_API_KEY          # 是否设置
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT SET'))"

# 修复:
cp config/.env.example config/.env
# 编辑 config/.env 填入真实 Key
```

### 15. 版本不兼容

```bash
# 检查版本
pip show langgraph langchain-core langchain-openai

# 推荐版本组合 (2025-2026):
# langgraph >= 0.2.0
# langchain-core >= 0.2.0
# langchain-openai >= 0.1.0
# pydantic >= 2.0.0

# 如果出现 ImportError，尝试:
pip install --upgrade langgraph langchain-core langchain-openai
```

### 16. `add_messages` 导入位置

```python
# ❌ 错误: 从 langchain_core 导入 (旧版)
from langchain_core.messages import add_messages

# ✅ 正确: 从 langgraph.graph 导入
from langgraph.graph import add_messages
```

---

## 七、调试技巧

### 打印图结构

```python
# 查看图的节点和边
app = graph.compile()
print(app.get_graph().draw_ascii())  # ASCII 图
# 或
print(app.get_graph().nodes)         # 节点列表
print(app.get_graph().edges)         # 边列表
```

### 用 debug_logs 追踪执行

```python
class State(TypedDict):
    # 加一个 debug 列表
    debug: Annotated[List[str], operator.add]

def my_node(state):
    # 每个节点都写日志
    return {"debug": [f"my_node 执行: input={state['data'][:20]}"]}
```

### 用 stream_mode="updates" 逐步查看

```python
for event in app.stream(input, stream_mode="updates"):
    for node, update in event.items():
        print(f"[{node}] → {list(update.keys())}")
```

---

## 速查表

| 问题 | 一句话修复 |
|------|-----------|
| 状态被覆盖 | 列表字段加 `Annotated[list, operator.add]` |
| interrupt 不工作 | 加 `checkpointer=MemorySaver()` |
| 没有记忆 | 传 `config={"configurable": {"thread_id": "xxx"}}` |
| 看不到 token 流 | 用 `stream_mode="messages"` 或 `astream_events` |
| 无限循环 | 加 `handoff_count` 计数器或 `max_iterations` |
| 子图事件缺失 | 加 `subgraphs=True` |
| 节点名找不到 | 检查 `add_node` 和 `add_edge` 中的拼写 |
| 重启丢记忆 | 换 `PostgresSaver` |
