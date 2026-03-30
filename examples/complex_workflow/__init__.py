"""
Phase 4-5 进阶 - 复杂工作流模式

学习目标:
- 多路条件分支 (一个路由函数 → 多个目标节点)
- 循环/迭代模式 (生成 → 评估 → 改进 → 收敛判断 → 循环或结束)
- 错误处理与重试 (异常捕获 → 恢复策略选择 → 条件重试)
- 防止无限循环 (max_iterations / max_attempts)

示例:
- conditional_flows.py     → 查询分类后路由到 4 个不同的处理节点
- loops_and_iteration.py   → 质量评估驱动的迭代优化循环
- error_handling.py        → 带退避策略的错误恢复工作流
"""
