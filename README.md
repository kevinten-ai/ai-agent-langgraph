# AI Agent LangGraph

> **从零到生产的 LangChain & LangGraph 完整学习路线**
>
> 图文文档 + 代码导读 + 8 个可运行示例 + 多 Agent 工作流系统

![LangChain vs LangGraph](./docs/images/04-langchain-vs-langgraph.png)

## 为什么学这个

大多数 AI Agent 框架（CrewAI、AutoGen、Dify）的底层逻辑，都可以映射到 LangGraph 的 **状态图 + 节点 + 条件边** 这套模型。掌握 LangChain + LangGraph，就掌握了 Agent 编排的底层范式。

```
LangChain  → 解决"调用 LLM"  → prompt | model | parser（线性管道）
LangGraph  → 解决"控制 LLM"  → StateGraph 有向图（循环 + 分支 + 持久化）
两者共享 langchain-core 基础设施，不是替代，而是从简单到复杂的连续谱
```

---

## 核心概念速览

<table>
<tr>
<td width="50%">

**LangChain 分层架构**

![LangChain Architecture](./docs/images/01-langchain-architecture.png)

</td>
<td width="50%">

**LCEL 管道组合**

![LCEL Pipeline](./docs/images/02-lcel-pipeline.png)

</td>
</tr>
<tr>
<td>

**LangGraph StateGraph**

![StateGraph](./docs/images/03-langgraph-stategraph.png)

</td>
<td>

**多 Agent Supervisor 模式**

![Supervisor](./docs/images/05-supervisor-pattern.png)

</td>
</tr>
</table>

<details>
<summary><b>RAG 检索增强生成</b>（点击展开）</summary>

![RAG Pipeline](./docs/images/06-rag-pipeline.png)

```
文档加载 → 文本分割 → 向量嵌入 → 存入向量库 → 检索 + LLM 生成回答
```

</details>

---

## 学习路线

```
基础                          进阶                           生产
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1-2         Phase 3        Phase 4-5          Phase 6
LangChain 核心     Agent+Tools    LangGraph 图编排     进阶 & 部署
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Runnable  │    │ @tool    │    │ State    │    │ interrupt()  │
│ LCEL      │───▶│ ReAct    │───▶│ Node     │───▶│ Checkpoint   │
│ Prompt    │    │ Agent    │    │ Edge     │    │ Streaming    │
│ RAG       │    │ 结构化输出│    │ 条件路由  │    │ Subgraph     │
└──────────┘    └──────────┘    └──────────┘    │ Swarm        │
                                                │ 部署          │
                                                └──────────────┘
```

### 学习文档

| # | 文档 | 内容 | 难度 |
|---|------|------|------|
| 00 | [学习路线总览](./docs/00-learning-roadmap.md) | 6 个 Phase 完整规划，模块-知识点-代码映射 | - |
| 01 | [LangChain 核心](./docs/01-langchain-core.md) | Runnable / LCEL / ChatModel / Prompt / RAG / Agent | 入门 |
| 02 | [LangGraph 核心](./docs/02-langgraph-core.md) | StateGraph / Node / Edge / Reducer / Checkpoint / 多Agent | 入门 |
| 03 | [架构总览](./docs/03-architecture-overview.md) | 两者关系、选型指南、本项目架构解析 | 入门 |
| 04 | [代码导读](./docs/04-code-guide.md) | 每个文件的关键行号标注，建议阅读顺序 | 入门 |
| 05 | [进阶专题](./docs/05-advanced-topics.md) | HITL / Checkpoint / Streaming / Subgraph / Swarm / 部署 | 进阶 |
| 06 | [FAQ & 踩坑指南](./docs/06-faq-and-pitfalls.md) | 17 个常见错误的症状、原因、修复 + 调试技巧速查表 | 全阶段 |

### Phase 1-3: LangChain 基础 → Agent

| 步骤 | 示例 | 你将学到 | 难度 |
|------|------|---------|------|
| 1 | [simple_chatbot.py](./examples/basic_agent/simple_chatbot.py) | StateGraph + Node + Edge 最小可运行图 | ★ |
| 2 | [agent_with_tools.py](./examples/basic_agent/agent_with_tools.py) | `@tool` + `bind_tools()` + 条件路由 | ★★ |

### Phase 4-5: LangGraph 图编排

| 步骤 | 示例 | 你将学到 | 难度 |
|------|------|---------|------|
| 3 | [conditional_flows.py](./examples/complex_workflow/conditional_flows.py) | 4 路条件分支 `add_conditional_edges` | ★★ |
| 4 | [loops_and_iteration.py](./examples/complex_workflow/loops_and_iteration.py) | 循环迭代 + 收敛检测 + 防无限循环 | ★★★ |
| 5 | [error_handling.py](./examples/complex_workflow/error_handling.py) | 错误恢复 + 重试 + 指数退避 | ★★★ |
| 6 | [role_based_agents.py](./examples/multi_agent/role_based_agents.py) | Supervisor 模式: Coordinator ↔ Workers 循环 | ★★★ |
| 7 | [message_passing.py](./examples/multi_agent/message_passing.py) | Agent 间消息总线通信 | ★★★ |
| 8 | [file_tools.py](./examples/mcp_integration/file_tools.py) | MCP 工具定义 + StateGraph 集成 | ★★★ |
| 9 | [demo.py](./demo.py) | 完整多 Agent 工作流系统 | ★★★★ |

### Phase 6: 进阶 → 生产 ([进阶专题文档](./docs/05-advanced-topics.md))

| 步骤 | 示例 | 你将学到 | 难度 |
|------|------|---------|------|
| 10 | [checkpoint_memory.py](./examples/advanced/checkpoint_memory.py) | MemorySaver 多轮记忆 + Thread 隔离 + 时间旅行 | ★★★ |
| 11 | [human_in_the_loop.py](./examples/advanced/human_in_the_loop.py) | `interrupt()` 暂停 + `Command(resume)` 恢复 + 审批流 | ★★★ |
| 12 | [streaming_output.py](./examples/advanced/streaming_output.py) | 4 种 stream_mode + `astream_events` 逐 token 流式 | ★★★ |
| 13 | [subgraph_composition.py](./examples/advanced/subgraph_composition.py) | 共享/不同 Schema 子图 + State 转换 + 子图流式事件 | ★★★★ |
| 14 | [swarm_agents.py](./examples/advanced/swarm_agents.py) | `Command(goto=)` Agent 间 handoff + 防无限循环 | ★★★★ |

> 遇到问题？查看 [FAQ & 踩坑指南](./docs/06-faq-and-pitfalls.md) — 17 个常见错误的症状、原因、一行修复

---

## 本项目：多 Agent 工作流系统

一个完整的 LangGraph 生产级示例，展示了条件路由、循环重试、MCP 工具集成等核心模式。

![本项目工作流](./docs/images/07-project-workflow.png)

```
START → task_assigner → tool_selector ─┬─ mcp_executor → executor → reviewer ─┬─ END
                                       │                                      │
                                       └─ executor (直接) ────────────────────│
                                                                              │
                                                       retry (< 6分且 < 2次) ─┘
```

| LangGraph 模式 | 代码位置 | 说明 |
|---------------|---------|------|
| State 定义 + 归约器 | [`src/models/states.py`](./src/models/states.py) | `Annotated[List, operator.add]` |
| StateGraph 构建 | [`src/workflow/orchestrator.py`](./src/workflow/orchestrator.py) | 5 节点 + 2 条件路由 |
| Agent 节点 | [`src/agents/task_assigner.py`](./src/agents/task_assigner.py) | LLM 任务分类 |
| MCP 工具链路 | [`src/mcp/`](./src/mcp/) | 注册 → 选择 → 并行执行 |

<details>
<summary><b>完整项目结构</b>（点击展开）</summary>

```
ai-agent-langgraph/
├── docs/                          # 学习文档 (7 篇 + 13 张图示)
│   ├── 00 ~ 05                    # 学习路线/核心概念/进阶专题
│   ├── 06-faq-and-pitfalls.md     # FAQ & 踩坑指南
│   └── images/                    # 13 张配图
├── src/                           # 多 Agent 工作流系统
│   ├── models/                    # State 定义 (base.py, states.py)
│   ├── agents/                    # Agent 节点 (task_assigner.py)
│   ├── mcp/                       # MCP 工具 (client, registry, selector, executor)
│   ├── utils/                     # 状态管理 (state_manager.py)
│   └── workflow/                  # 图编排 (orchestrator.py)
├── examples/                      # 13 个可运行示例
│   ├── basic_agent/               # 基础 Agent (chatbot, tools)
│   ├── multi_agent/               # 多 Agent (supervisor, message passing)
│   ├── complex_workflow/          # 复杂工作流 (条件/循环/错误处理)
│   ├── mcp_integration/           # MCP 集成 (file tools)
│   └── advanced/                  # 进阶 (checkpoint/HITL/streaming/subgraph/swarm)
├── config/                        # 配置 (env.example)
├── demo.py                        # 完整系统演示
└── requirements.txt               # 依赖 (带中文注释)
```

</details>

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/kevinten-ai/ai-agent-langgraph.git
cd ai-agent-langgraph

# 2. 环境准备
python -m venv venv && source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp config/.env.example config/.env
# 编辑 config/.env，填入 OPENAI_API_KEY

# 5. 运行第一个示例
python examples/basic_agent/simple_chatbot.py

# 6. 运行完整系统演示
python demo.py
```

## 技术栈

| 依赖 | 用途 |
|------|------|
| `langgraph` | 核心: StateGraph, 条件路由, Checkpoint |
| `langchain-core` | 基础: Runnable, Messages, Tools |
| `langchain-openai` | LLM: ChatOpenAI |
| `langchain-community` | 社区集成: DocumentLoaders, VectorStores |
| `pydantic` | 状态模型定义与验证 |

## 学习资源

- [LangChain Academy: Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) — 官方免费课程
- [LangChain 文档](https://python.langchain.com/) / [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph) / [MCP 协议](https://modelcontextprotocol.io/)

## License

MIT
