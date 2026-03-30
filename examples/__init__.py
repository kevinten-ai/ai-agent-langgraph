"""
LangGraph 学习示例

按照学习顺序排列:

1. basic_agent/        - LangGraph 基础
   - simple_chatbot.py        → 最小 StateGraph: State + Node + Edge
   - agent_with_tools.py      → 工具调用 + 条件路由 (add_conditional_edges)

2. multi_agent/        - 多 Agent 协作
   - role_based_agents.py     → Supervisor 模式: coordinator 分配任务给 worker
   - message_passing.py       → Agent 间消息传递机制

3. complex_workflow/   - 复杂工作流
   - conditional_flows.py     → 多路条件分支 (4 种查询类型 → 4 条路径)
   - loops_and_iteration.py   → 循环迭代: 生成 → 评估 → 改进 → 收敛检查
   - error_handling.py        → 错误处理 + 重试 + 退避策略

4. mcp_integration/    - MCP 工具集成
   - file_tools.py            → MCP 协议工具定义与图集成
"""
