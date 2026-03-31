"""
Phase 6 - LangGraph 进阶示例

学习目标:
- Checkpoint 持久化与多轮记忆
- interrupt() 人机交互审批流
- 流式输出 (逐 token / 逐节点)
- 子图嵌套组合
- Swarm 多 Agent 自主交接

示例:
- checkpoint_memory.py     → MemorySaver 实现跨轮对话记忆 + 时间旅行
- human_in_the_loop.py     → interrupt() 暂停 + Command(resume) 恢复
- streaming_output.py      → stream_mode 四种模式 + 逐 token 输出
- subgraph_composition.py  → 父图嵌套子图 + State 转换
- swarm_agents.py          → Agent 间 handoff 自主交接
"""
