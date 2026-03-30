# LangGraph 核心概念

> LangChain 解决"调用 LLM"的问题，LangGraph 解决"控制 LLM"的问题。用有向图取代线性链，节点做事，边决定下一步去哪。

## 一、为什么需要 LangGraph

LangChain 的 LCEL 是**线性管道** —— 数据从左流到右：

```
prompt → model → parser     （LCEL：单向，无法回头）
```

但真实的 Agent 需要：
- **循环**：模型调用工具后，需要回到模型再次思考
- **分支**：根据不同条件走不同路径
- **等待**：暂停执行，等待人类输入
- **持久化**：保存状态，跨会话恢复

LangGraph 用**有向图**解决这些问题：

```
         ┌──────────┐
         │  START   │
         └────┬─────┘
              │
         ┌────▼─────┐
    ┌───►│  Agent   │◄────┐
    │    └────┬─────┘     │
    │         │           │
    │    ┌────▼─────┐     │
    │    │ 路由函数  │     │
    │    └──┬───┬───┘     │
    │  "工具"│   │"结束"   │
    │       │   │         │
    │  ┌────▼─┐ │         │
    │  │ Tool ├─┘         │
    │  └──┬───┘           │
    │     └───────────────┘
    │
         ┌──────────┐
         │   END    │
         └──────────┘
```

> LangGraph 的灵感来自 Google Pregel 和 Apache Beam，采用了 NetworkX 风格的 API。

---

## 二、核心概念

### 1. State — 共享状态

State 是图中所有节点共享的数据结构，是 LangGraph 的核心：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

# 方式一：TypedDict（轻量）
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 带归约器
    next_step: str                            # 默认覆盖

# 方式二：Pydantic（完整验证）— 本项目使用的方式
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    user_input: str = Field("", description="用户输入")
    task_type: Optional[TaskType] = None
    debug_logs: Annotated[List[str], operator.add] = Field(default_factory=list)
```

### 归约器 (Reducer)

归约器决定**当多个节点更新同一字段时，如何合并**：

```python
# 无归约器 → 后写入的覆盖先写入的
next_step: str                              # 覆盖语义

# operator.add 归约器 → 列表追加
debug_logs: Annotated[List[str], operator.add]  # [a] + [b] = [a, b]

# add_messages 归约器 → 智能消息合并（按 ID 追踪）
messages: Annotated[list, add_messages]     # 追加新消息，按 ID 更新旧消息
```

**本项目的 `src/models/states.py` 示例：**

```python
class AgentState(BaseModel):
    # 普通字段 — 覆盖语义
    user_input: str = Field("", description="用户原始输入")
    task_type: Optional[TaskType] = None
    review_score: Optional[float] = None

    # 归约器字段 — 追加语义
    debug_logs: Annotated[List[str], operator.add] = Field(default_factory=list)
    error_messages: Annotated[List[str], operator.add] = Field(default_factory=list)
```

---

### 2. Node — 节点（做事的）

节点是 Python 函数，接收当前状态，返回状态更新：

```python
def agent_node(state: State) -> dict:
    """节点的签名：接收 State → 返回部分更新"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # 只返回要更新的字段

async def async_node(state: State) -> dict:
    """也支持异步"""
    result = await some_api_call(state["query"])
    return {"result": result}
```

**本项目中的节点实现（`src/workflow/orchestrator.py`）：**

```python
async def _task_assigner_node(self, state: AgentState) -> AgentState:
    """任务分配节点"""
    updated_state = await self.task_assigner.assign_task(state)
    updated_state.debug_logs.append(f"任务分配完成: {updated_state.task_type}")
    return updated_state

async def _executor_node(self, state: AgentState) -> AgentState:
    """执行节点 — 整合信息并生成最终回答"""
    final_answer = await self._generate_final_answer(state)
    state.final_answer = final_answer
    return state
```

---

### 3. Edge — 边（决定流向的）

边连接节点，决定执行顺序：

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

# 添加节点
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("reviewer", reviewer_node)

# ① 固定边 — 总是从 A 到 B
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")     # 工具执行完回到 agent

# ② 条件边 — 根据状态动态路由
def route_function(state: State) -> str:
    """返回下一个节点的名称"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", route_function)

# ③ 条件边 + 映射表
graph.add_conditional_edges(
    "reviewer",
    should_retry,          # 返回 "retry" 或 "end"
    {
        "retry": "agent",  # 映射到实际节点
        "end": END
    }
)
```

**本项目中的条件路由：**

```python
# MCP 工具选择路由
graph.add_conditional_edges(
    "tool_selector",
    self._should_use_mcp,       # 返回 "mcp" 或 "direct"
    {"mcp": "mcp_executor", "direct": "executor"}
)

# 审核重试路由
graph.add_conditional_edges(
    "reviewer",
    self._should_retry,         # 返回 "retry" 或 "end"
    {"retry": "executor", "end": END}
)
```

---

### 4. 编译与执行

```python
# 编译图 — 生成可执行应用
app = graph.compile()

# 执行
result = app.invoke({"messages": [HumanMessage("你好")]})

# 异步执行
result = await app.ainvoke(initial_state)

# 流式执行
for event in app.stream(initial_state):
    print(event)
```

---

### 5. Checkpointer — 状态持久化

每个 super-step 自动保存状态快照：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 开发环境：内存
checkpointer = MemorySaver()

# 生产环境：SQLite / PostgreSQL
# checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = graph.compile(checkpointer=checkpointer)

# 通过 thread_id 维持会话
config = {"configurable": {"thread_id": "user-123"}}

# 第一轮对话
result1 = app.invoke({"messages": [HumanMessage("我叫张三")]}, config=config)

# 第二轮对话 — 自动恢复之前的状态
result2 = app.invoke({"messages": [HumanMessage("我叫什么？")]}, config=config)
# → "你叫张三"
```

**Checkpoint 支持的能力：**

| 能力 | 说明 |
|------|------|
| 持久化记忆 | 通过 thread_id 跨会话保持状态 |
| 故障恢复 | 从中断处精确恢复执行 |
| 时间旅行 | 回放和检视历史状态 |

---

## 三、执行模型

LangGraph 使用**消息传递**模型，执行分为离散的 **super-step**：

1. 节点初始为不活跃状态
2. 收到消息后激活，执行计算
3. 发送消息到下游节点
4. 所有节点不活跃且无消息传输时，执行终止

默认递归限制：1000 个 super-step。可通过 `RemainingSteps` 在接近限制时优雅降级。

---

## 四、进阶特性

### 1. Human-in-the-Loop (人机交互)

两种机制：

```python
# ① interrupt() — 动态中断（推荐）
from langgraph.types import interrupt

def human_approval_node(state: State) -> dict:
    """在任何位置动态暂停"""
    answer = interrupt("请确认是否继续执行？")  # 暂停，等待人类输入
    if answer == "yes":
        return {"approved": True}
    return {"approved": False}

# ② Breakpoints — 静态中断（在编译时指定）
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_node"],   # 执行前暂停
    interrupt_after=["review_node"],       # 执行后暂停
)
```

**恢复执行：**

```python
# 图暂停后，通过 Command 恢复
from langgraph.types import Command

# 传入人类的决定，恢复执行
app.invoke(Command(resume="yes"), config=config)
```

---

### 2. 多智能体模式

#### Supervisor 模式（本项目采用）

一个中央调度 Agent 分配任务给 Worker：

```python
from langgraph_supervisor import create_supervisor

# 创建 supervisor
supervisor = create_supervisor(
    model=ChatOpenAI(model="gpt-4o"),
    agents=[researcher, writer, reviewer],
    prompt="你是项目经理，根据任务分配给合适的团队成员"
)
app = supervisor.compile()
```

**本项目的 Supervisor 实现（`orchestrator.py`）：**

```
START → task_assigner → tool_selector → [mcp_executor/executor] → reviewer → END
                                                                      │
                                                                      └── retry → executor (循环)
```

#### Swarm 模式（对等通信）

多个 Agent 直接通信，无中央调度：

```python
# Agent A 完成后，根据结果决定交给 Agent B 还是 Agent C
def agent_a(state):
    result = model.invoke(state["messages"])
    if "需要数据分析" in result.content:
        return Command(goto="analyst")
    return Command(goto="writer")
```

#### Pipeline 模式（顺序执行）

```
Agent A → Agent B → Agent C → 输出
（研究员）  （写手）    （审核员）
```

---

### 3. 子图 (Subgraph)

将复杂图拆分为可复用的子图：

```python
# 定义子图
sub_graph = StateGraph(SubState)
sub_graph.add_node(...)
sub_app = sub_graph.compile()

# 嵌入主图
main_graph = StateGraph(MainState)
main_graph.add_node("sub_process", sub_app)  # 子图作为节点
```

---

### 4. 流式输出

```python
# 基础流式
for event in app.stream(input):
    for node_name, output in event.items():
        print(f"[{node_name}]: {output}")

# 细粒度事件流
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        # 逐 token
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_tool_start":
        print(f"\n工具调用: {event['name']}")
```

---

### 5. Send 对象（Map-Reduce）

动态创建并行分支：

```python
from langgraph.types import Send

def router(state):
    """为每个任务创建一个并行分支"""
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

graph.add_conditional_edges("planner", router)
```

---

### 6. 节点缓存

```python
from langgraph.types import CachePolicy

# TTL 缓存 — 相同输入在 60 秒内返回缓存结果
@graph.add_node
def expensive_node(state):
    ...

graph.nodes["expensive_node"].cache_policy = CachePolicy(ttl=60)
```

---

## 五、完整示例：ReAct Agent

把所有概念组合成一个完整的 Agent：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 定义工具
@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query} 相关信息..."

# 2. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. 定义节点
model = ChatOpenAI(model="gpt-4o").bind_tools([search])

def agent(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def tool_executor(state: State):
    # 执行最后一条消息中的工具调用
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        result = search.invoke(tc["args"])
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": results}

# 4. 定义路由
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. 构建图
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", tool_executor)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # 工具执行完回到 agent

# 6. 编译（带持久化）
app = graph.compile(checkpointer=MemorySaver())

# 7. 执行
config = {"configurable": {"thread_id": "demo"}}
result = app.invoke(
    {"messages": [("human", "搜索一下 LangGraph 是什么")]},
    config=config
)
```
