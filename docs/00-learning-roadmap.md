# LangChain & LangGraph 学习路线

> 从核心抽象到多智能体系统的系统化学习路线

## 总览

```
Phase 1          Phase 2         Phase 3          Phase 4          Phase 5          Phase 6
LangChain 核心    RAG 检索增强     LangChain Agent   LangGraph 基础    LangGraph 进阶    实战项目
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Runnable  │    │ Document │    │ Tools    │    │ State    │    │ HITL     │    │ Chatbot  │
│ ChatModel │───▶│ Embed    │───▶│ Agent    │───▶│ Node     │───▶│ Multi-   │───▶│ RAG App  │
│ LCEL     │    │ Vector   │    │ Struct   │    │ Edge     │    │  Agent   │    │ Multi-   │
│ Prompt   │    │ Retrieve │    │ Output   │    │ Check    │    │ SubGraph │    │  Agent   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Phase 1 — LangChain 核心基础

> 📖 详细文档：[LangChain 核心概念](./01-langchain-core.md)

| 模块 | 主题 | 核心知识点 | 对应示例 |
|------|------|-----------|---------|
| 01-setup | 环境搭建 | pip install, API Key 配置, 包结构理解 | - |
| 02-runnable | Runnable 接口 | invoke/stream/batch, 统一协议, Schema | - |
| 03-chat-models | 模型调用 | ChatOpenAI/ChatAnthropic, 消息类型, binding | `examples/basic_agent/simple_chatbot.py` |
| 04-prompts | 提示词模板 | ChatPromptTemplate, 变量注入, Few-shot | - |
| 05-output-parsers | 输出解析器 | JsonOutputParser, PydanticOutputParser | - |
| 06-lcel | LCEL 链式组合 | pipe `\|`, RunnableParallel, Passthrough, Lambda | - |
| 07-callbacks | 回调与追踪 | CallbackHandler, LangSmith, 可观测性 | - |

## Phase 2 — RAG 检索增强生成

> 📖 详细文档：[LangChain 核心概念 - RAG 部分](./01-langchain-core.md#rag-检索增强生成)

| 模块 | 主题 | 核心知识点 |
|------|------|-----------|
| 01-document-loaders | 文档加载 | PDF/Web/CSV Loader, Document 对象 |
| 02-text-splitters | 文本分割 | RecursiveCharacterTextSplitter, 分割策略 |
| 03-embeddings | 向量嵌入 | OpenAIEmbeddings, 嵌入原理, 相似度搜索 |
| 04-vector-stores | 向量数据库 | FAISS, Chroma, Pinecone 集成 |
| 05-retrieval-chain | 检索链 | BaseRetriever, create_retrieval_chain |

## Phase 3 — LangChain Agents

> 📖 详细文档：[LangChain 核心概念 - Agent 部分](./01-langchain-core.md#agent-智能体)

| 模块 | 主题 | 核心知识点 | 对应示例 |
|------|------|-----------|---------|
| 01-tools | 工具定义 | @tool 装饰器, BaseTool, 输入 Schema | `examples/basic_agent/agent_with_tools.py` |
| 02-create-agent | Agent 创建 | create_agent(), ReAct 模式, Tool Calling | `examples/multi_agent/` |
| 03-middleware | 中间件 | HITL 审批中间件, 消息摘要中间件 | - |
| 04-structured-output | 结构化输出 | with_structured_output(), Pydantic 绑定 | - |

## Phase 4 — LangGraph 基础

> 📖 详细文档：[LangGraph 核心概念](./02-langgraph-core.md)

| 模块 | 主题 | 核心知识点 | 对应代码 |
|------|------|-----------|---------|
| 01-state-graph | 状态图 | StateGraph, TypedDict/Pydantic, compile() | `src/models/states.py` |
| 02-nodes-edges | 节点与边 | add_node(), add_edge(), START/END | `src/workflow/orchestrator.py` |
| 03-conditional | 条件路由 | add_conditional_edges(), Command | `src/workflow/orchestrator.py` |
| 04-reducers | 状态归约 | Annotated 归约器, add_messages, operator.add | `src/models/states.py` |
| 05-checkpoint | 状态持久化 | MemorySaver, SqliteSaver, thread_id | - |

## Phase 5 — LangGraph 进阶

> 📖 详细文档：[LangGraph 核心概念 - 进阶部分](./02-langgraph-core.md#进阶特性)

| 模块 | 主题 | 核心知识点 | 对应代码 |
|------|------|-----------|---------|
| 01-hitl | 人机交互 | interrupt(), breakpoints, 审批流程 | - |
| 02-supervisor | 监督者模式 | create_supervisor(), 中央调度 | `src/workflow/orchestrator.py` |
| 03-swarm | 群体模式 | 对等通信, 去中心化协作 | - |
| 04-subgraphs | 子图 | 图嵌套, 状态映射, 模块化组合 | - |
| 05-streaming | 流式输出 | astream_events(), 事件过滤 | - |

## Phase 6 — 实战项目

| 项目 | 描述 | 综合知识点 |
|------|------|-----------|
| 01-chatbot | 多轮对话机器人 | Memory + Streaming + HITL |
| 02-rag-app | 文档问答系统 | RAG 全链路 + Agent 工具集成 |
| 03-multi-agent | 多智能体协作系统 | Supervisor + Subgraph + Checkpoint |

## 本项目代码与学习路线的映射

```
本项目 src/                          对应学习阶段
├── models/
│   ├── base.py                     ← Phase 4: 状态定义基础
│   └── states.py                   ← Phase 4: AgentState + 归约器
├── agents/
│   └── task_assigner.py            ← Phase 3: Agent + Tools
├── mcp/
│   ├── client.py                   ← Phase 3: 工具集成
│   ├── registry.py                 ← Phase 3: 工具注册
│   ├── selector.py                 ← Phase 3: 工具选择
│   └── executor.py                 ← Phase 3: 工具执行
├── utils/
│   └── state_manager.py            ← Phase 4: 状态持久化
└── workflow/
    └── orchestrator.py             ← Phase 4+5: StateGraph + 条件路由 + 多Agent
```

## 进阶专题

> 详见 [进阶专题](./05-advanced-topics.md)，覆盖生产级 Agent 的 6 大必备能力。

| 专题 | 核心能力 | 难度 |
|------|---------|------|
| Human-in-the-Loop | `interrupt()` 动态中断 + `Command(resume=...)` 恢复 | 中 |
| Checkpoint & Memory | MemorySaver/PostgresSaver + Store 跨线程记忆 | 中 |
| Streaming | stream_mode + astream_events 逐 token 流式 | 中 |
| Subgraph | 子图嵌套 + State 转换 + 中断冒泡 | 高 |
| Advanced Multi-Agent | Command/Send/Swarm 模式 | 高 |
| Deployment | langgraph.json + Server API + 生产部署 | 高 |

## 代码阅读顺序

> 详见 [代码导读](./04-code-guide.md)，含每个文件的关键行号标注。

```
入门:
  simple_chatbot.py → agent_with_tools.py → conditional_flows.py

进阶:
  loops_and_iteration.py → role_based_agents.py → error_handling.py

深入:
  src/models/states.py → src/workflow/orchestrator.py → src/agents/ → src/mcp/
```

## 推荐学习资源

| 资源 | 说明 |
|------|------|
| [LangChain Academy: Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) | 官方免费课程 |
| [LangChain 官方文档](https://python.langchain.com/) | Python SDK 文档 |
| [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | LangGraph 文档 |
| [MCP 协议规范](https://modelcontextprotocol.io/) | Model Context Protocol |
