"""
Phase 5 入门 - 多 Agent 协作

学习目标:
- Supervisor 模式 (协调者 + 多个 Worker)
- Agent 间消息传递
- 条件循环: Worker 完成后回到 Coordinator 分配下一个任务
- 状态中存储多个 Agent 的结果

示例:
- role_based_agents.py  → 经典 Supervisor 模式: Coordinator → Researcher → Analyst → Reporter
- message_passing.py    → 消息总线模式: Agent 通过结构化消息通信
"""
