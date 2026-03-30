# LangGraph 进阶专题

> 你已经掌握了 StateGraph / Node / Edge / 条件路由的基础。本文档覆盖 6 个进阶主题，每个都是生产级 Agent 的必备能力。

---

## 一、Human-in-the-Loop (人机交互)

> 让 Agent 在关键步骤暂停，等待人类审批、编辑或输入后再继续。

### 1.1 两种机制对比

| 特性 | `interrupt()` (推荐) | Breakpoints |
|------|---------------------|-------------|
| 定义位置 | 代码中任意位置 | `compile()` 时静态指定 |
| 是否可条件触发 | 是，可以 `if` 判断后触发 | 否，总是触发 |
| 传递信息 | 可传递提示信息给人类 | 无 |
| 适用场景 | 动态审批、用户确认 | 固定断点调试 |

### 1.2 interrupt() — 动态中断

```python
from langgraph.types import interrupt, Command

def human_approval_node(state):
    """在执行危险操作前请求人类确认"""
    if state["action_type"] == "delete":
        # 暂停执行，向人类展示提示信息
        human_response = interrupt(
            "即将删除数据，确认继续？请回复 yes/no"
        )
        if human_response != "yes":
            return {"action_cancelled": True}

    # 继续执行
    return {"action_cancelled": False}
```

**工作原理：**
1. `interrupt()` 被调用时，LangGraph **保存当前状态到 Checkpointer**
2. 图执行暂停，控制权返回给调用方
3. 调用方获取人类输入后，通过 `Command(resume=...)` 恢复执行
4. `interrupt()` 返回人类提供的值，节点继续执行

### 1.3 恢复执行

```python
from langgraph.types import Command

# 图暂停后，获取人类输入
user_input = input("请确认 (yes/no): ")

# 恢复执行 — resume 的值会成为 interrupt() 的返回值
result = app.invoke(
    Command(resume=user_input),
    config={"configurable": {"thread_id": "thread-1"}}
)
```

### 1.4 Breakpoints — 静态断点

```python
# 在 compile 时指定断点
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["dangerous_node"],   # 执行前暂停
    interrupt_after=["review_node"],       # 执行后暂停
)

# 执行到断点会自动暂停
result = app.invoke(input, config)
# result 是断点处的状态

# 检查状态后恢复
result = app.invoke(None, config)  # 传 None 继续执行
```

### 1.5 实用模式

```python
# 模式 1: 审批流 — 人类审批后才执行工具
def tool_approval(state):
    tool_calls = state["messages"][-1].tool_calls
    approval = interrupt(f"Agent 要调用以下工具:\n{tool_calls}\n确认？")
    if approval == "reject":
        return {"messages": [AIMessage(content="操作已取消")]}
    # 执行工具...

# 模式 2: 编辑 Agent 输出 — 人类可以修改 Agent 的草稿
def draft_and_edit(state):
    draft = llm.invoke(state["messages"])
    edited = interrupt(f"Agent 草稿:\n{draft.content}\n请编辑或直接回车接受:")
    final = edited if edited else draft.content
    return {"messages": [AIMessage(content=final)]}

# 模式 3: 多步确认 — 每步都可以中断
def step_by_step(state):
    for step in state["plan"]:
        confirm = interrupt(f"即将执行: {step}\n继续？")
        if confirm != "yes":
            return {"stopped_at": step}
        execute(step)
```

> **关键前提：** `interrupt()` 必须配合 Checkpointer 使用，否则无法保存和恢复状态。

---

## 二、Checkpointing 与 Memory

> Checkpoint 是 LangGraph 的"存档点"机制，让图执行可以暂停、恢复、回滚。

### 2.1 Checkpointer 类型

| Checkpointer | 适用场景 | 持久化 |
|--------------|---------|--------|
| `MemorySaver` | 开发/测试 | 内存，进程退出即丢失 |
| `SqliteSaver` | 单机生产 | SQLite 文件 |
| `PostgresSaver` | 分布式生产 | PostgreSQL |

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# 开发环境
checkpointer = MemorySaver()

# 单机生产
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 分布式生产
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@host:5432/db"
)

app = graph.compile(checkpointer=checkpointer)
```

### 2.2 Thread 管理

```python
# 每个 thread_id 是一个独立的会话
config_user_a = {"configurable": {"thread_id": "user-alice-001"}}
config_user_b = {"configurable": {"thread_id": "user-bob-002"}}

# 同一 thread 内的多次调用共享状态
app.invoke({"messages": [HumanMessage("我叫张三")]}, config_user_a)
app.invoke({"messages": [HumanMessage("我叫什么？")]}, config_user_a)
# → "你叫张三"（记住了上下文）

# 不同 thread 完全隔离
app.invoke({"messages": [HumanMessage("我叫什么？")]}, config_user_b)
# → "我不知道你叫什么"
```

### 2.3 时间旅行 (Time Travel)

```python
# 获取状态历史 — 查看每个 super-step 的快照
history = list(app.get_state_history(config))
for state in history:
    print(f"Step: {state.metadata['step']}")
    print(f"  Node: {state.metadata.get('source', 'N/A')}")
    print(f"  State: {state.values}")

# 回滚到某个历史状态
target_state = history[3]  # 第 4 个快照
# 从该状态重新执行
result = app.invoke(None, target_state.config)
```

### 2.4 跨 Thread 长期记忆 (Store API)

Checkpointer 的记忆是**线程内**的。如果需要**跨线程**的长期记忆（如用户偏好），使用 Store：

```python
from langgraph.store.memory import InMemoryStore
# 生产环境: from langgraph.store.postgres import PostgresStore

store = InMemoryStore()
app = graph.compile(checkpointer=checkpointer, store=store)

# 在节点中使用 Store
def my_node(state, config, *, store):
    user_id = config["configurable"]["user_id"]

    # 读取长期记忆
    memories = store.search(("user_preferences", user_id))

    # 写入长期记忆
    store.put(("user_preferences", user_id), "lang", {"value": "zh-CN"})

    return state
```

**记忆类型总结：**

| 类型 | 作用域 | 机制 | 用途 |
|------|--------|------|------|
| State (messages) | 单次图执行 | State 字段 | 对话上下文 |
| Checkpoint | 单个 Thread | Checkpointer | 会话历史、暂停恢复 |
| Store | 跨 Thread | Store API | 用户偏好、长期知识 |

---

## 三、Streaming (流式输出)

> 让用户实时看到 Agent 的思考过程和输出，而不是等到全部完成。

### 3.1 Stream Modes

LangGraph 支持多种流式模式，可以**同时使用多种**：

| Mode | 输出内容 | 适用场景 |
|------|---------|---------|
| `"values"` | 每步后的**完整** State | 调试，需要看到完整状态 |
| `"updates"` | 每步的**增量**更新 | 生产，只关心变化的部分 |
| `"messages"` | 逐 token 的消息流 | 聊天 UI，实时显示文字 |
| `"events"` | 细粒度内部事件 | 高级：追踪工具调用、嵌套 Agent |

### 3.2 基础流式

```python
# 流式获取每步更新
for event in app.stream(input, config, stream_mode="updates"):
    for node_name, update in event.items():
        print(f"[{node_name}] 更新: {update}")

# 流式获取完整状态
for event in app.stream(input, config, stream_mode="values"):
    print(f"当前状态: {event}")
```

### 3.3 逐 Token 流式 (聊天 UI 必备)

```python
# 方式 1: stream_mode="messages" — 最简单
for msg_chunk, metadata in app.stream(
    input, config, stream_mode="messages"
):
    if hasattr(msg_chunk, "content") and msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)

# 方式 2: astream_events — 最细粒度
async for event in app.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        # 逐 token
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n🔧 调用工具: {event['name']}")
    elif event["event"] == "on_tool_end":
        print(f"✅ 工具返回: {event['data'].output[:100]}")
```

### 3.4 多模式组合

```python
# 同时流式输出 updates 和 messages
for event in app.stream(
    input, config,
    stream_mode=["updates", "messages"],
    subgraphs=True  # 包含子图的流式输出
):
    mode, data = event
    if mode == "messages":
        chunk, metadata = data
        print(chunk.content, end="")
    elif mode == "updates":
        print(f"\n--- 节点更新: {data} ---")
```

### 3.5 区分 stream vs astream_events

```
stream(stream_mode="messages")  → 状态级别的消息流，适合简单聊天
astream_events()                → 执行级别的事件流，适合复杂 Agent
                                  可以捕获嵌套工具调用、子 Agent 输出等
```

---

## 四、Subgraph (子图)

> 将复杂图拆分为可复用的模块，像函数一样组合。

### 4.1 何时使用子图

- 图变得太大，需要模块化
- 某个子流程需要在多个图中复用
- 不同团队开发不同模块
- 需要不同的状态 Schema

### 4.2 共享 Schema — 直接作为节点

当子图和父图有**相同的 State key** 时，直接嵌入：

```python
# 子图定义
sub_builder = StateGraph(SharedState)
sub_builder.add_node("step_a", step_a_fn)
sub_builder.add_node("step_b", step_b_fn)
sub_builder.add_edge(START, "step_a")
sub_builder.add_edge("step_a", "step_b")
sub_builder.add_edge("step_b", END)
sub_graph = sub_builder.compile()

# 父图中直接作为节点
parent_builder = StateGraph(SharedState)
parent_builder.add_node("pre_process", pre_fn)
parent_builder.add_node("sub_process", sub_graph)  # 子图作为节点
parent_builder.add_node("post_process", post_fn)
parent_builder.add_edge(START, "pre_process")
parent_builder.add_edge("pre_process", "sub_process")
parent_builder.add_edge("sub_process", "post_process")
parent_builder.add_edge("post_process", END)
```

### 4.3 不同 Schema — 函数包装转换

当子图和父图有**不同的 State** 时，需要手动转换：

```python
class ParentState(TypedDict):
    user_query: str
    final_answer: str

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]
    result: str

# 子图
child_graph = StateGraph(ChildState)
# ... 构建子图 ...
child_app = child_graph.compile()

# 包装函数：负责 State 转换
def child_node(state: ParentState) -> dict:
    # 父 State → 子 State
    child_input = {
        "messages": [HumanMessage(content=state["user_query"])],
        "result": ""
    }
    # 调用子图
    child_result = child_app.invoke(child_input)
    # 子 State → 父 State
    return {"final_answer": child_result["result"]}

# 父图中使用包装函数
parent_builder.add_node("child", child_node)
```

### 4.4 子图中的中断

子图中的 `interrupt()` 会**向上冒泡**到父图：

```python
# 子图中有 interrupt
def child_approval(state):
    answer = interrupt("子图请求确认")
    return {"approved": answer == "yes"}

# 恢复时，对父图调用 Command(resume=...)
# resume 值会自动传递到子图的 interrupt
result = parent_app.invoke(
    Command(resume="yes"),
    config
)
```

### 4.5 关键规则

- 子图**继承**父图的 Checkpointer（不需要单独编译时传入）
- 子图有**独立的状态空间**，不会污染父图
- 流式输出时需要 `subgraphs=True` 才能看到子图内部事件
- 子图可以嵌套子图（祖父 → 父 → 子）

---

## 五、Advanced Multi-Agent (高级多智能体)

### 5.1 Command — 节点级路由控制

`Command` 让节点**同时更新状态和控制流向**：

```python
from langgraph.types import Command

def smart_router(state) -> Command:
    """节点直接决定下一步去哪"""
    if "数据分析" in state["task"]:
        return Command(
            update={"routed_to": "analyst"},  # 更新状态
            goto="analyst_node"                # 跳转到指定节点
        )
    elif "写报告" in state["task"]:
        return Command(
            update={"routed_to": "writer"},
            goto="writer_node"
        )
    return Command(goto=END)
```

**vs `add_conditional_edges` 的区别：**

| 方式 | 路由逻辑位置 | 能否同时更新状态 |
|------|------------|----------------|
| `add_conditional_edges` | 图构建时，外部路由函数 | 否 |
| `Command(goto=...)` | 节点内部 | 是 |

### 5.2 Send — 动态并行 (Map-Reduce)

`Send` 为每个子任务创建独立的并行分支：

```python
from langgraph.types import Send

def planner(state):
    """将大任务拆分为子任务，并行分发"""
    subtasks = state["subtasks"]  # ["分析市场", "分析技术", "分析竞品"]

    # 为每个子任务创建一个 Send，并行执行
    return [
        Send("worker", {"task": subtask, "result": ""})
        for subtask in subtasks
    ]

def worker(state):
    """处理单个子任务"""
    result = llm.invoke(f"执行任务: {state['task']}")
    return {"result": result.content}

def aggregator(state):
    """汇总所有子任务结果"""
    all_results = state["results"]  # 归约器自动收集
    summary = llm.invoke(f"汇总以下结果:\n{all_results}")
    return {"final_summary": summary.content}

# 图构建
graph.add_node("planner", planner)
graph.add_node("worker", worker)
graph.add_node("aggregator", aggregator)
graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", lambda _: "worker")  # Send 自动路由
graph.add_edge("worker", "aggregator")
graph.add_edge("aggregator", END)
```

### 5.3 Swarm 模式 — Agent 自主交接

Agent 之间直接交接控制权，无需中央调度：

```python
# 使用官方 langgraph-swarm 库
# pip install langgraph-swarm
from langgraph_swarm import create_swarm, create_handoff_tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# 定义 Agent
math_agent = create_react_agent(
    model,
    tools=[calculator, create_handoff_tool(agent_name="writer")],
    name="math_expert",
    prompt="你是数学专家。如果需要写报告，交给 writer。"
)

writer_agent = create_react_agent(
    model,
    tools=[write_doc, create_handoff_tool(agent_name="math_expert")],
    name="writer",
    prompt="你是写作专家。如果需要计算，交给 math_expert。"
)

# 创建 Swarm
swarm = create_swarm(
    agents=[math_agent, writer_agent],
    default_agent="math_expert"
)
app = swarm.compile(checkpointer=MemorySaver())
```

**Handoff 工具原理：**

```python
# create_handoff_tool 内部生成类似这样的工具
@tool
def transfer_to_writer(task_description: str, state: Annotated[dict, InjectedState]):
    """将任务转交给 writer Agent"""
    return Command(
        update={"active_agent": "writer"},
        goto="writer"
    )
```

### 5.4 三种多 Agent 模式对比

| 模式 | 控制方式 | 适用场景 | 复杂度 |
|------|---------|---------|--------|
| **Supervisor** | 中央调度 | 任务分工明确，需要全局协调 | 中 |
| **Swarm** | Agent 自主交接 | 对话式，Agent 间灵活协作 | 低 |
| **Pipeline** | 固定顺序 | 线性处理流水线 | 最低 |

---

## 六、部署 (LangGraph Platform)

### 6.1 本地开发服务器

```bash
# 安装
pip install langgraph-cli

# 启动开发服务器
langgraph dev
# → http://localhost:8123
# 内置 API 文档: http://localhost:8123/docs
```

### 6.2 langgraph.json 配置

```json
{
  "graphs": {
    "my_agent": "./src/agent.py:graph",
    "rag_agent": "./src/rag.py:rag_graph"
  },
  "dependencies": ["./requirements.txt"],
  "env": ".env"
}
```

- `graphs`: 键是 assistant 名称，值是 `文件路径:变量名`
- 每个 graph entry 成为一个独立的 API 端点

### 6.3 API 端点

LangGraph Server 提供 30+ REST API：

```bash
# 创建 thread
curl -X POST http://localhost:8123/threads

# 发送消息并流式获取回复
curl -X POST http://localhost:8123/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "my_agent", "input": {"messages": [{"role": "human", "content": "你好"}]}}'

# 获取 thread 状态
curl http://localhost:8123/threads/{thread_id}/state

# 获取状态历史 (时间旅行)
curl http://localhost:8123/threads/{thread_id}/history
```

### 6.4 生产部署选项

| 方式 | 特点 |
|------|------|
| **LangSmith Deployment (Cloud SaaS)** | 代码放 GitHub，平台自动构建部署，含 PostgreSQL |
| **Self-Hosted** | Docker 部署到自己的服务器 |
| **langgraph dev** | 仅开发/测试用，内存模式 |

```bash
# Docker 部署
langgraph build -t my-agent-image
docker run -p 8123:8000 my-agent-image
```

---

## 七、进阶状态管理

### 7.1 Input/Output Schema 分离

限制图的外部接口，隐藏内部状态：

```python
class InputState(TypedDict):
    user_query: str

class OutputState(TypedDict):
    final_answer: str

class InternalState(InputState, OutputState):
    # 这些字段外部不可见
    intermediate_results: list
    agent_scratchpad: str
    retry_count: int

# 构建时指定三种 Schema
graph = StateGraph(
    InternalState,
    input=InputState,    # invoke 时只接受这些字段
    output=OutputState   # 返回时只暴露这些字段
)
```

### 7.2 RemainingSteps — 优雅降级

```python
from langgraph.types import RemainingSteps

class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps  # 自动管理

def agent_node(state):
    if state["remaining_steps"] < 3:
        # 快接近递归限制了，直接给出最终回答
        return {"messages": [AIMessage(content="已接近步骤限制，这是当前最佳答案...")]}
    # 正常执行...
```

---

## 学习建议

按以下顺序实践：

```
1. Checkpoint + Thread    → 给 simple_chatbot.py 加上 MemorySaver，实现多轮记忆
2. interrupt()            → 给 agent_with_tools.py 加上工具审批流
3. Streaming              → 给任意示例加上逐 token 流式输出
4. Subgraph               → 把 orchestrator.py 的审核流程拆为子图
5. Command + Send         → 实现一个 Map-Reduce 并行任务分发
6. Swarm                  → 用 langgraph-swarm 构建自主协作的 Agent 团队
```

## 参考资源

- [LangGraph Interrupts 官方文档](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Streaming 官方文档](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph Subgraphs 官方文档](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [LangGraph Persistence 官方文档](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Command: Multi-Agent 新工具 (官方博客)](https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/)
- [LangGraph Swarm 库](https://github.com/langchain-ai/langgraph-swarm-py)
- [LangGraph 本地服务器](https://docs.langchain.com/oss/python/langgraph/local-server)
- [LangGraph Platform API](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html)
- [LangGraph 长期记忆指南](https://markaicode.com/langgraph-memory-short-term-long-term-storage/)
