# 代码导读

> 按学习顺序，标注每个文件的关键代码位置和学习重点。

## 一、从 examples 入门

### 1. `examples/basic_agent/simple_chatbot.py` — 最小 LangGraph 应用

```
重点关注:
├── L26-31   ChatbotState(TypedDict) — State 定义，Annotated[List, operator.add] 归约器
├── L33-51   process_input() — Node 实现：接收 State，返回更新
├── L54-82   generate_response() — Node 实现：调用 ChatOpenAI
├── L85-105  create_chatbot_graph() — 图构建的完整流程：
│            ├── StateGraph(ChatbotState) — 创建图
│            ├── add_node() — 添加节点
│            ├── add_edge(START, ...) — 添加边
│            └── graph.compile() — 编译
└── L136     chatbot.invoke({...}) — 执行图
```

**学到的概念:** StateGraph, TypedDict State, Node, Edge, compile(), invoke()

---

### 2. `examples/basic_agent/agent_with_tools.py` — 工具 + 条件路由

```
重点关注:
├── L37-53   @tool calculator — 用 @tool 装饰器定义工具
├── L56-94   @tool web_search — 带默认参数的工具
├── L117-148 decide_next_action() — 模型绑定工具: llm.bind_tools([...])
│            └── 检查 response.tool_calls 是否有工具调用
├── L151-194 execute_tools() — 执行工具并创建 ToolMessage
├── L225-238 should_use_tools() — 条件路由函数: 返回节点名称字符串
└── L261-268 add_conditional_edges() — 条件边: 路由函数 → 节点映射
```

**学到的概念:** @tool, bind_tools(), tool_calls, ToolMessage, add_conditional_edges()

---

### 3. `examples/multi_agent/role_based_agents.py` — Supervisor 模式

```
重点关注:
├── L36-125  三个专业 Agent 类: ResearchAgent, AnalysisAgent, ReportAgent
├── L128-158 coordinator_agent() — Supervisor 节点: 根据状态决定分配哪个 Worker
├── L201-218 should_continue() — 条件路由: 判断 research/analyze/report 哪步还没完成
├── L237-256 create_multi_agent_graph() — 图构建:
│            ├── coordinator → 条件边 → research/analyze/report
│            ├── research → coordinator (循环回来)
│            ├── analyze → coordinator (循环回来)
│            └── report → END
└── 关键模式: coordinator ←→ worker 的循环协作
```

**学到的概念:** Supervisor 模式, 循环图结构, 多 Agent 状态共享

---

### 4. `examples/complex_workflow/conditional_flows.py` — 多路条件分支

```
重点关注:
├── L35-95   analyze_query() — 用 LLM 分类查询类型
├── L98-208  四个处理节点: factual/analytical/creative/unknown
├── L211-230 route_by_query_type() — 条件路由: 返回 4 种节点名称
└── L274-283 add_conditional_edges() — 4 路分支:
             {"factual_handler", "analytical_handler", "creative_handler", "unknown_handler"}
```

**学到的概念:** 多路条件分支, 路由函数返回字符串映射

---

### 5. `examples/complex_workflow/loops_and_iteration.py` — 迭代循环

```
重点关注:
├── L52-97   content_generation_step() — 根据迭代次数调整策略
├── L100-155 quality_evaluation_step() — 质量评估
├── L205-240 check_convergence() — 收敛判断: 4 个停止条件
│            ├── 最大迭代次数
│            ├── 评分达标 (>= 8.5)
│            ├── 评分收敛 (变化 < 阈值)
│            └── 质量未改善
├── L273-291 should_iterate_again() — 条件边: "content_generation" 或 "finalize"
└── L318-325 条件边形成循环:
             check_convergence → content_generation (循环) 或 finalize (退出)
```

**学到的概念:** 循环模式, 收敛检测, 防止无限循环

---

## 二、深入 src 核心代码

### 6. `src/models/states.py` — State 定义 (核心)

```
重点关注:
├── L17-60   AgentState(BaseModel) — 完整的 Pydantic State 定义
│            ├── L21    user_input: str — 普通字段 (覆盖语义)
│            ├── L54    debug_logs: Annotated[List[str], operator.add] — 归约器字段 (追加语义)
│            └── L55    error_messages: Annotated[List[str], operator.add] — 归约器字段
├── L63-131  子状态类: TaskAssignerState, ExecutorState, ReviewerState 等
├── L134-145 create_initial_state() — 工厂函数
└── L148-211 状态转换辅助函数: update_state_with_*
```

**学到的概念:** Pydantic State vs TypedDict, Annotated 归约器, 状态转换函数

---

### 7. `src/workflow/orchestrator.py` — StateGraph 编排 (核心)

```
重点关注:
├── L58-59   StateGraph(AgentState) — 创建图
├── L64-103  _build_workflow() — 完整的图构建:
│            ├── L67-71  add_node() × 5 — 添加 5 个节点
│            ├── L74     set_entry_point() — 设置入口
│            ├── L81-88  add_conditional_edges("tool_selector", ...) — MCP 路由
│            └── L96-103 add_conditional_edges("reviewer", ...) — 重试路由
├── L62      self.app = self.workflow.compile() — 编译
├── L123     await self.app.ainvoke(initial_state) — 异步执行
├── L367-369 _should_use_mcp() — 条件路由函数: 返回 "mcp" 或 "direct"
└── L371-373 _should_retry() — 条件路由函数: 返回 "retry" 或 "end"
```

**学到的概念:** 完整的 StateGraph 生命周期, 多条件路由, 异步执行

---

### 8. `src/agents/task_assigner.py` — Agent 节点实现

```
重点关注:
├── L12      from langchain_openai import ChatOpenAI — LangChain 模型
├── L34      self.llm = ChatOpenAI(model="gpt-4o-mini") — 模型初始化
├── L101-136 analyze_task() — 任务分析: 分类 → 优先级 → 复杂度
├── L170-201 _llm_classify_task_type() — 用 LLM 做分类 (Prompt → invoke)
└── L407-451 assign_task(state) — 节点函数: 接收 State → 返回 State
```

**学到的概念:** ChatOpenAI 使用, Prompt 构建, Agent 节点的输入输出

---

### 9. `src/mcp/` — MCP 工具集成

```
registry.py  — 工具注册表 (注册/查询/索引)
├── L35-310  _initialize_builtin_tools() — 6 个内置工具定义
└── L387-402 get_tools_for_task() — 按任务类型查找工具

selector.py  — 智能工具选择
├── L84-156  select_tools_for_task() — 4 种选择策略组合
└── L158-185 _calculate_task_relevance() — 相关性评分算法

executor.py  — 并行执行
├── L47-110  execute_parallel_primary() — asyncio.gather 并行执行
└── L168-193 execute_with_fallback() — 降级方案

client.py    — HTTP 客户端
├── L22-70   ConnectionPool — 连接池
├── L73-161  CacheManager — 缓存管理
└── L256-301 _call_tool_with_retry() — 指数退避重试
```

**学到的概念:** 工具注册/发现/选择/执行的完整链路, 异步并行, 缓存和重试

---

## 三、建议阅读顺序

```
入门:
  simple_chatbot.py → agent_with_tools.py → conditional_flows.py

进阶:
  loops_and_iteration.py → role_based_agents.py → error_handling.py

深入:
  src/models/states.py → src/workflow/orchestrator.py → src/agents/ → src/mcp/
```
