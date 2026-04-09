# Agent Platform 学习路线

> 从核心抽象到多智能体系统，再到生产级 Agent Platform 的完整学习路线

## 总览

```
Phase 1          Phase 2         Phase 3          Phase 4          Phase 5          Phase 6            Phase 7              Phase 8            Phase 9
LangChain 核心    RAG 检索增强     LangChain Agent   LangGraph 基础    LangGraph 进阶    实战项目             可观测性              评估体系             生产部署
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Runnable  │    │ Document │    │ Tools    │    │ State    │    │ HITL     │    │ Chatbot  │    │ LangSmith    │    │ RAGAS        │    │ langgraph    │
│ ChatModel │───▶│ Embed    │───▶│ Agent    │───▶│ Node     │───▶│ Multi-   │───▶│ RAG App  │───▶│ OpenTelemetry│───▶│ Custom Eval  │───▶│ Docker / K8s │
│ LCEL     │    │ Vector   │    │ Struct   │    │ Edge     │    │  Agent   │    │ Multi-   │    │ Node Metrics │    │ CI Dataset   │    │ Server API   │
│ Prompt   │    │ Retrieve │    │ Output   │    │ Check    │    │ SubGraph │    │  Agent   │    │ Structured   │    │ Threshold    │    │ Cloud        │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    │ Logs         │    │ Pipeline     │    │              │
                                                                                                 └──────────────┘    └──────────────┘    └──────────────┘
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

## Phase 7 — Agent 可观测性

> 📖 详细文档：[Agent 可观测性](./07-agent-observability.md)

| 模块 | 主题 | 核心知识点 | 对应示例 |
|------|------|-----------|---------|
| 01-langsmith | 自动追踪 | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, Run Tree | `examples/platform/langsmith_tracing.py` |
| 02-astream-events | 结构化日志 | `astream_events`, 节点耗时/Token 统计 | `examples/advanced/streaming_output.py` |
| 03-opentelemetry | OpenTelemetry 集成 | `TracerProvider`, `ConsoleSpanExporter`, 手动埋点 | `examples/platform/opentelemetry_tracing.py` |
| 04-custom-metrics | 自定义指标 | 节点级吞吐/延迟/错误率统计 | `examples/platform/opentelemetry_tracing.py` |

## Phase 8 — Agent 评估体系

> 📖 详细文档：[Agent 评估体系](./08-agent-evaluation.md)

| 模块 | 主题 | 核心知识点 | 对应示例 |
|------|------|-----------|---------|
| 01-ragas | 端到端 RAG 评估 | `faithfulness`, `answer_relevancy`, `context_relevancy` | `examples/platform/ragas_evaluation.py` |
| 02-custom-eval | 自定义 Eval Pipeline | 轨迹评分、工具正确性、JSON 格式、引用检测 | `examples/platform/custom_eval_pipeline.py` |
| 03-node-assertions | 节点级断言 | State 中间状态校验、边界条件测试 | `tests/` (pytest) |
| 04-ci-integration | CI 集成 | 数据集 + 阈值 + 回归测试 |

## Phase 9 — 生产部署

> 📖 详细文档：[生产部署实战](./09-production-deployment.md)

| 模块 | 主题 | 核心知识点 | 对应配置 |
|------|------|-----------|---------|
| 01-langgraph-json | 服务入口 | `langgraph.json` 配置，`get_graph()` 工厂函数 | `langgraph.json` |
| 02-local-dev | 本地开发 | `langgraph dev`, `langgraph up`, Docker Desktop | `Dockerfile` |
| 03-self-host | 自托管 | Docker Compose (API + Postgres + Redis), 健康检查 | `docker-compose.yml` |
| 04-cloud | 云平台 | LangGraph Cloud, Server API, 自动扩缩容 | — |

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

平台能力示例 examples/platform/       对应学习阶段
├── langsmith_tracing.py            ← Phase 7: LangSmith 自动追踪
├── opentelemetry_tracing.py        ← Phase 7: OpenTelemetry 手动埋点
├── ragas_evaluation.py             ← Phase 8: RAGAS 自动评估
├── custom_eval_pipeline.py         ← Phase 8: 自定义 Eval Pipeline
└── langgraph_server/               ← Phase 9: LangGraph Server 部署配置
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

## Agent Platform 能力矩阵

> 从零到生产，Agent 系统还需要以下平台级能力：

| 专题 | 文档 | 核心能力 | 难度 |
|------|------|---------|------|
| Observability | [07-agent-observability.md](./07-agent-observability.md) | LangSmith 自动追踪 + OpenTelemetry 手动埋点 + Node Metrics | 中 |
| Evaluation | [08-agent-evaluation.md](./08-agent-evaluation.md) | RAGAS 指标 + 自定义 Eval Pipeline + CI 数据集回归 | 中高 |
| Production Deployment | [09-production-deployment.md](./09-production-deployment.md) | `langgraph dev` / Docker Compose 自托管 / LangGraph Cloud | 中高 |


## 代码阅读顺序

> 详见 [代码导读](./04-code-guide.md)，含每个文件的关键行号标注。

```
入门:
  simple_chatbot.py → agent_with_tools.py → conditional_flows.py

进阶:
  loops_and_iteration.py → role_based_agents.py → error_handling.py

深入:
  src/models/states.py → src/workflow/orchestrator.py → src/agents/ → src/mcp/

平台能力:
  langsmith_tracing.py → opentelemetry_tracing.py → ragas_evaluation.py → custom_eval_pipeline.py → langgraph_server/
```

## 推荐学习资源

| 资源 | 说明 |
|------|------|
| [LangChain Academy: Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) | 官方免费课程 |
| [LangChain 官方文档](https://python.langchain.com/) | Python SDK 文档 |
| [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | LangGraph 文档 |
| [MCP 协议规范](https://modelcontextprotocol.io/) | Model Context Protocol |
