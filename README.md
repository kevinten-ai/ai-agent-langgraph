# AI Agent LangGraph 学习项目

这是一个基于LangGraph框架的AI Agent学习项目，从基础概念到高级应用的完整实现指南。

## 📚 项目结构

```
ai-agent-langgraph/
├── README.md                    # 项目说明
├── requirements.txt            # 项目依赖
├── demo.py                     # 系统演示脚本
├── src/                        # 源代码
│   ├── __init__.py            # 包初始化
│   ├── models/                # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py           # 基础模型定义
│   │   └── states.py         # 状态模型定义
│   ├── agents/                # Agent实现
│   │   ├── __init__.py
│   │   └── task_assigner.py  # 任务分配Agent
│   ├── mcp/                   # MCP集成
│   │   ├── __init__.py
│   │   ├── client.py         # MCP客户端
│   │   ├── registry.py       # 工具注册表
│   │   ├── selector.py       # 工具选择器
│   │   └── executor.py       # 工具执行器
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   └── state_manager.py  # 状态管理器
│   └── workflow/              # 工作流编排
│       ├── __init__.py
│       └── orchestrator.py   # 工作流编排器
├── examples/                   # 示例代码
│   ├── basic_agent/           # 基础Agent示例
│   ├── multi_agent/           # 多Agent协作
│   ├── complex_workflow/      # 复杂工作流
│   └── mcp_integration/       # MCP工具集成
├── config/                     # 配置文件
│   ├── env.example           # 环境变量示例
│   └── README.md             # 配置说明
└── tests/                      # 测试代码
```

## 🎯 核心概念

### 多Agent协作工作流
项目实现了完整的多Agent协作系统，支持任务分析、工具选择、并行执行和结果审核。

```python
from src import MultiAgentWorkflow

# 创建多Agent工作流
workflow = MultiAgentWorkflow(enable_mcp_integration=True)

# 执行任务
result = await workflow.run_workflow("分析当前AI发展趋势")
print(result["final_answer"])
```

### MCP工具集成
基于Model Context Protocol的标准化工具集成，支持动态工具发现和调用。

```python
from src.mcp import MCPToolRegistry, ToolSelector, MCPToolExecutor

# 注册和选择工具
registry = MCPToolRegistry()
selector = ToolSelector(registry)

tools = selector.select_tools_for_task(
    task_type=TaskType.DATA_ANALYSIS,
    user_input="分析销售数据"
)
```

### 状态图 (State Graph)
LangGraph使用有向图来表示Agent的工作流，每个节点是一个处理步骤，边表示状态转换。

```python
from langgraph.graph import StateGraph, START, END

# 定义状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("task_assigner", task_assigner_node)
workflow.add_node("executor", executor_node)
workflow.add_edge(START, "task_assigner")
workflow.add_edge("task_assigner", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()
result = app.invoke(initial_state)
```

## 🚀 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
cp config/.env.example config/.env
# 编辑 .env 文件，添加你的API密钥
```

### 运行完整系统演示

```bash
# 运行完整的多Agent协作演示
python demo.py
```

### 运行单个Agent示例

```bash
# 基础聊天Agent
cd examples/basic_agent
python simple_chatbot.py

# 带工具的Agent
python agent_with_tools.py
```

## 📖 学习路径

### 1. 基础Agent (Basic Agents)
- [简单问答Agent](./examples/basic_agent/simple_chatbot.py)
- [带工具的Agent](./examples/basic_agent/agent_with_tools.py)
- [状态管理](./examples/basic_agent/state_management.py)

### 2. 多Agent协作 (Multi-Agent)
- [角色分工系统](./examples/multi_agent/role_based_agents.py)
- [消息传递机制](./examples/multi_agent/message_passing.py)
- [任务分配](./examples/multi_agent/task_delegation.py)

### 3. 复杂工作流 (Complex Workflows)
- [条件分支](./examples/complex_workflow/conditional_flows.py)
- [循环处理](./examples/complex_workflow/loops_and_iteration.py)
- [错误处理](./examples/complex_workflow/error_handling.py)

### 4. MCP集成 (MCP Integration)
- [文件系统工具](./examples/mcp_integration/file_tools.py)
- [网络请求工具](./examples/mcp_integration/web_tools.py)
- [数据库工具](./examples/mcp_integration/database_tools.py)

## 🎯 核心概念

### 状态图 (State Graph)
LangGraph使用有向图来表示Agent的工作流，每个节点是一个处理步骤，边表示状态转换。

```python
from langgraph.graph import StateGraph, START, END

# 定义状态
class AgentState(TypedDict):
    messages: list
    current_step: str

# 创建图
graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
```

### 检查点 (Checkpoints)
自动保存和恢复Agent的执行状态，支持长时间运行的任务。

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 运行时指定线程ID
config = {"configurable": {"thread_id": "session_1"}}
result = app.invoke(inputs, config=config)
```

### 人机交互 (Human-in-the-Loop)
允许在执行过程中暂停并等待人工输入。

```python
from langgraph.types import interrupt

def human_approval(state):
    user_input = interrupt("需要人工确认，是否继续？")
    return {"approved": user_input == "yes"}
```

## 🔧 开发工具

### 核心依赖
- `langgraph`: 核心框架
- `langchain-openai`: OpenAI集成
- `langchain-core`: LangChain核心
- `python-dotenv`: 环境变量管理

### 开发工具
- `pytest`: 测试框架
- `black`: 代码格式化
- `mypy`: 类型检查
- `pre-commit`: 代码质量检查

## 🧪 测试运行

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_basic_agent.py

# 运行带覆盖率的测试
pytest --cov=src --cov-report=html
```

## 📚 学习资源

- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain文档](https://python.langchain.com/)
- [AI Agent设计模式](https://www.patterns.app/)
- [MCP协议规范](https://modelcontextprotocol.io/)

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 🙏 致谢

- [LangGraph团队](https://github.com/langchain-ai/langgraph) - 提供了优秀的框架
- [Anthropic](https://www.anthropic.com/) - MCP协议的提出者
- 所有为AI Agent发展做出贡献的开发者们