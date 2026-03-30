# 架构总览：LangChain 与 LangGraph 的关系

## 一、全景图

```
┌──────────────────────────────────────────────────────────────┐
│                       你的 AI 应用                            │
│                                                              │
│   简单场景                           复杂场景                  │
│   ┌──────────────┐                 ┌─────────────────┐       │
│   │  LangChain   │                 │   LangGraph     │       │
│   │  LCEL Chain  │                 │   StateGraph    │       │
│   │  A → B → C   │                 │   有向图编排     │       │
│   │  (线性管道)   │                 │   (循环+分支)    │       │
│   └──────────────┘                 └─────────────────┘       │
│          │                                │                  │
│          └────────────┬───────────────────┘                  │
│                       │                                      │
│                 共享基础设施                                    │
│   ┌──────────────────────────────────────────────┐           │
│   │              langchain-core                   │           │
│   │   Runnable │ LCEL │ ChatModel │ Tools │       │           │
│   │   Retriever │ Messages │ OutputParser         │           │
│   └──────────────────────────────────────────────┘           │
│                       │                                      │
│   ┌──────────────────────────────────────────────┐           │
│   │        Provider 集成 (可互换)                  │           │
│   │  langchain-openai │ langchain-anthropic │ ... │           │
│   └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

## 二、对比

| 维度 | LangChain (LCEL) | LangGraph |
|------|-------------------|-----------|
| **定位** | 高级 API，快速构建 | 低级编排，精细控制 |
| **数据流** | 线性管道 `A → B → C` | 有向图（循环、分支、并行） |
| **状态管理** | 无内置状态 | 一等公民：TypedDict + 归约器 |
| **持久化** | 无 | Checkpointer（内存/SQLite/PostgreSQL） |
| **人机交互** | 通过中间件 | 原生 `interrupt()` 支持 |
| **多 Agent** | 需手动编排 | Supervisor / Swarm / Pipeline 模式 |
| **典型场景** | 问答链、RAG、简单 Agent | 复杂 Agent、多智能体、审批流 |
| **上手难度** | 低，几行代码跑通 | 中，需理解图论思维 |
| **调试** | print / callback | 可视化图 + 时间旅行回放 |

## 三、什么时候用哪个

### 用 LangChain LCEL

```python
# ✅ 简单的 prompt → model → parse 链
chain = prompt | model | parser

# ✅ RAG 检索问答
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

# ✅ 带工具的简单 Agent（内部自动用 LangGraph）
agent = create_agent(model, tools)
```

**适合场景：**
- 一问一答，不需要复杂控制流
- 标准 RAG 管道
- 简单的 ReAct Agent（用 `create_agent`）

### 用 LangGraph

```python
# ✅ 需要条件分支
graph.add_conditional_edges("analyzer", route_to_specialist)

# ✅ 需要循环重试
graph.add_conditional_edges("reviewer", should_retry, {"retry": "executor", "end": END})

# ✅ 需要人工审批
def approval(state):
    answer = interrupt("确认发送邮件？")
    ...

# ✅ 多 Agent 协作
supervisor = create_supervisor(model, [agent_a, agent_b, agent_c])
```

**适合场景：**
- 执行流需要循环（重试、迭代优化）
- 需要条件分支（根据结果走不同路径）
- 需要人机交互（暂停等待审批）
- 多 Agent 协作
- 长时间运行的任务（需要持久化）

### 递进关系

```
复杂度   用什么
─────   ─────────────────────────────────
  低    LCEL chain: prompt | model | parser
  ↓     create_agent(model, tools)
  ↓     StateGraph 单 Agent + 条件路由
  ↓     StateGraph + Checkpoint + HITL
  高    多 Agent Supervisor + Subgraph
```

> **不是替代关系，而是从简单到复杂的连续谱。** 大多数 AI Agent 框架（CrewAI、AutoGen）的底层，都可以映射到 LangGraph 的"状态图 + 节点 + 条件边"这套模型。

## 四、本项目架构解析

本项目 (`ai-agent-langgraph`) 是一个完整的 **LangGraph 多 Agent 系统**：

### 工作流图

![本项目工作流](./images/07-project-workflow.png)

```
START
  │
  ▼
┌─────────────────┐
│  task_assigner   │  分析用户输入 → 识别任务类型和优先级
│  (任务分配)       │  使用 LLM 理解意图
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  tool_selector   │  根据任务类型 → 从注册表选择合适的 MCP 工具
│  (工具选择)       │  最多选 3 个工具
└────────┬────────┘
         │
    ┌────▼─────┐
    │ 条件路由  │  _should_use_mcp()
    └──┬────┬──┘  有工具 → "mcp" / 无工具 → "direct"
  "mcp"│    │"direct"
       │    │
  ┌────▼──┐ │
  │ mcp   │ │     并行执行 MCP 工具调用
  │ exec  ├─┘     收集工具返回结果
  └───────┘
       │
       ▼
┌─────────────────┐
│    executor      │  整合所有信息（用户输入 + 任务分析 + MCP 结果）
│    (执行)        │  使用 LLM 生成最终答案
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    reviewer      │  评分（0-10）：基于执行状态、错误数、重试次数
│    (审核)        │  < 6 分且重试 < 2 次 → 重试
└────────┬────────┘
         │
    ┌────▼─────┐
    │ 条件路由  │  _should_retry()
    └──┬────┬──┘
 "retry"│    │"end"
        │    │
        │    ▼
        │   END ✅
        │
        └──▶ executor (循环重试)
```

### 状态流转

```
initialized → task_analyzed → tool_selected → executed → reviewed → completed
                                                  │                    ▲
                                                  └── retry ──────────┘
```

### 核心设计模式映射

| 本项目组件 | LangGraph 模式 | 说明 |
|-----------|----------------|------|
| `AgentState` (Pydantic) | State Schema | 完整的状态定义，带归约器 |
| `_build_workflow()` | Graph Construction | 声明式图构建 |
| `_should_use_mcp()` | Conditional Edge | 根据状态动态路由 |
| `_should_retry()` | Loop Pattern | 条件循环（审核不通过则重试） |
| `StateManager` | State Persistence | 工作流状态管理 |
| `MCPToolRegistry` | Tool Registration | 动态工具注册与发现 |
| `TaskAssigner` → `Executor` | Supervisor Pattern | 任务分配 → 专业执行 |

## 五、技术栈依赖关系

```
本项目依赖
├── langgraph >= 0.2.0          # 核心：StateGraph, START, END
├── langchain-openai >= 0.1.0   # LLM：ChatOpenAI
├── langchain-core >= 0.2.0     # 基础：Messages, Tools, Runnable
├── langchain-community >= 0.2.0 # 社区集成
├── pydantic >= 2.0.0           # 状态模型定义
├── python-dotenv >= 1.0.0      # 环境变量
├── aiohttp >= 3.9.0            # MCP 异步 HTTP
├── faiss-cpu >= 1.7.0          # 向量检索（可选）
└── chromadb >= 0.4.0           # 向量数据库（可选）
```

## 六、学习建议

1. **先跑通 `examples/basic_agent/simple_chatbot.py`**，理解最小可用的 LangGraph Agent
2. **阅读 `src/models/states.py`**，理解状态定义和归约器
3. **阅读 `src/workflow/orchestrator.py`**，理解图的构建和条件路由
4. **修改条件路由逻辑**，体验 LangGraph 的灵活性
5. **添加新节点**，比如增加一个"摘要生成"节点
6. **集成 Checkpointer**，体验持久化和时间旅行
