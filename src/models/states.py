"""
状态模型定义

定义LangGraph工作流的状态模型，支持复杂的状态流转和数据传递。
"""

from typing import Annotated, List, Optional, Dict, Any
import operator
from pydantic import BaseModel, Field
from .base import (
    Task, TaskType, TaskPriority, ExecutionStatus,
    MCPToolDefinition, MCPToolSelection, MCPExecutionResult,
    ExecutionResult, ReviewResult, WorkflowMetadata
)


class AgentState(BaseModel):
    """多Agent工作流状态定义"""

    # 用户输入
    user_input: str = Field("", description="用户原始输入")

    # 任务分析结果
    task: Optional[Task] = Field(None, description="解析后的任务对象")
    task_type: Optional[TaskType] = Field(None, description="任务类型")
    task_description: Optional[str] = Field(None, description="任务描述")
    task_priority: Optional[TaskPriority] = Field(None, description="任务优先级")

    # MCP工具相关
    available_tools: List[MCPToolDefinition] = Field(default_factory=list, description="可用工具列表")
    selected_tools: List[MCPToolSelection] = Field(default_factory=list, description="选择的工具")
    tool_recommendations: Dict[str, Any] = Field(default_factory=dict, description="工具推荐信息")
    mcp_results: List[MCPExecutionResult] = Field(default_factory=list, description="MCP执行结果")

    # 执行结果
    execution_result: Optional[ExecutionResult] = Field(None, description="执行结果")
    execution_status: Optional[ExecutionStatus] = Field(None, description="执行状态")

    # 审核结果
    review_score: Optional[float] = Field(None, description="审核评分")
    review_feedback: Optional[str] = Field(None, description="审核反馈")
    needs_retry: Optional[bool] = Field(None, description="是否需要重试")
    retry_count: int = Field(0, description="重试次数")

    # 控制流
    current_agent: Optional[str] = Field(None, description="当前活跃Agent")
    workflow_status: str = Field("initialized", description="工作流状态")
    next_step: Optional[str] = Field(None, description="下一步操作")

    # 工作流元数据
    workflow_metadata: WorkflowMetadata = Field(default_factory=WorkflowMetadata, description="工作流元数据")

    # 调试和日志
    debug_logs: Annotated[List[str], operator.add] = Field(default_factory=list, description="调试日志")
    error_messages: Annotated[List[str], operator.add] = Field(default_factory=list, description="错误消息")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="性能指标")

    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True


class TaskAssignerState(AgentState):
    """TaskAssigner Agent状态"""

    # 任务分析特定字段
    analysis_confidence: float = Field(0.0, description="分析置信度")
    alternative_task_types: List[TaskType] = Field(default_factory=list, description="备选任务类型")
    task_complexity_score: float = Field(0.0, description="任务复杂度评分")


class ExecutorState(AgentState):
    """Executor Agent状态"""

    # 执行特定字段
    execution_plan: List[Dict[str, Any]] = Field(default_factory=list, description="执行计划")
    parallel_execution: bool = Field(False, description="是否并行执行")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")


class ReviewerState(AgentState):
    """Reviewer Agent状态"""

    # 审核特定字段
    quality_checks: List[Dict[str, Any]] = Field(default_factory=list, description="质量检查结果")
    improvement_suggestions: List[str] = Field(default_factory=list, description="改进建议")
    review_criteria: Dict[str, float] = Field(default_factory=dict, description="审核标准")


class MCPClientState(BaseModel):
    """MCP客户端状态"""

    server_url: str = Field(..., description="MCP服务器URL")
    connection_status: str = Field("disconnected", description="连接状态")
    last_heartbeat: float = Field(0.0, description="最后心跳时间")
    active_requests: int = Field(0, description="活跃请求数")
    total_requests: int = Field(0, description="总请求数")
    error_count: int = Field(0, description="错误计数")
    response_times: List[float] = Field(default_factory=list, description="响应时间历史")


class CacheState(BaseModel):
    """缓存状态"""

    cache_size: int = Field(0, description="缓存项数量")
    memory_usage: float = Field(0.0, description="内存使用(MB)")
    hit_rate: float = Field(0.0, description="缓存命中率")
    eviction_count: int = Field(0, description="驱逐计数")
    last_cleanup: float = Field(0.0, description="最后清理时间")


class MonitoringState(BaseModel):
    """监控状态"""

    active_workflows: int = Field(0, description="活跃工作流数")
    completed_workflows: int = Field(0, description="完成工作流数")
    failed_workflows: int = Field(0, description="失败工作流数")
    average_execution_time: float = Field(0.0, description="平均执行时间")
    error_rate: float = Field(0.0, description="错误率")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="资源使用情况")


class ErrorRecoveryState(BaseModel):
    """错误恢复状态"""

    current_strategy: str = Field("none", description="当前恢复策略")
    retry_attempts: int = Field(0, description="重试次数")
    last_error: Optional[str] = Field(None, description="最后错误")
    recovery_actions: List[Dict[str, Any]] = Field(default_factory=list, description="恢复行动记录")
    degraded_mode: bool = Field(False, description="降级模式")


# 状态转换辅助函数
def create_initial_state(user_input: str) -> AgentState:
    """创建初始状态"""
    from datetime import datetime

    return AgentState(
        user_input=user_input,
        workflow_metadata=WorkflowMetadata(
            workflow_id=f"wf_{int(datetime.now().timestamp())}",
            start_time=datetime.now()
        ),
        debug_logs=[f"初始化工作流，用户输入: {user_input[:100]}..."]
    )


def update_state_with_task_analysis(state: AgentState, task: Task) -> AgentState:
    """使用任务分析结果更新状态"""
    state.task = task
    state.task_type = task.task_type
    state.task_description = task.task_description
    state.task_priority = task.priority
    state.current_agent = "task_assigner"
    state.workflow_status = "task_analyzed"
    state.debug_logs.append(f"任务分析完成: {task.task_type.value}")

    return state


def update_state_with_tool_selection(state: AgentState, selected_tools: List[MCPToolSelection]) -> AgentState:
    """使用工具选择结果更新状态"""
    state.selected_tools = selected_tools
    state.debug_logs.append(f"选择工具: {[t.tool_name for t in selected_tools]}")

    return state


def update_state_with_execution_result(state: AgentState, execution_result: ExecutionResult) -> AgentState:
    """使用执行结果更新状态"""
    state.execution_result = execution_result
    state.execution_status = execution_result.status
    state.current_agent = "executor"
    state.workflow_status = "executed"
    state.debug_logs.append(f"执行完成，状态: {execution_result.status.value}")

    return state


def update_state_with_review_result(state: AgentState, review_result: ReviewResult) -> AgentState:
    """使用审核结果更新状态"""
    state.review_score = review_result.overall_score
    state.review_feedback = review_result.feedback
    state.needs_retry = review_result.needs_retry
    state.current_agent = "reviewer"
    state.workflow_status = "reviewed"
    state.debug_logs.append(f"审核完成，评分: {review_result.overall_score}")

    return state


def finalize_state(state: AgentState, final_answer: str) -> AgentState:
    """完成状态"""
    from datetime import datetime

    state.workflow_metadata.end_time = datetime.now()
    state.workflow_metadata.duration = (
        state.workflow_metadata.end_time - state.workflow_metadata.start_time
    ).total_seconds()
    state.workflow_status = "completed"
    state.debug_logs.append("工作流完成")

    # 创建最终的ExecutionResult
    if not state.execution_result:
        state.execution_result = ExecutionResult(
            content=final_answer,
            status=ExecutionStatus.SUCCESS
        )

    return state


# 状态验证函数
def validate_state_transition(from_state: AgentState, to_state: AgentState, transition: str) -> bool:
    """
    验证状态转换的合法性

    Args:
        from_state: 起始状态
        to_state: 目标状态
        transition: 转换名称

    Returns:
        是否合法
    """
    transition_rules = {
        "task_assigner->executor": lambda f, t: (
            t.task_type is not None and
            t.task_description is not None
        ),
        "executor->reviewer": lambda f, t: (
            t.execution_result is not None and
            t.execution_status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.PARTIAL]
        ),
        "reviewer->end": lambda f, t: (
            t.review_score is not None and
            not t.needs_retry
        )
    }

    validator = transition_rules.get(transition)
    if validator:
        return validator(from_state, to_state)

    return True


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """获取状态摘要"""
    return {
        "workflow_id": state.workflow_metadata.workflow_id,
        "status": state.workflow_status,
        "current_agent": state.current_agent,
        "task_type": state.task_type.value if state.task_type else None,
        "execution_status": state.execution_status.value if state.execution_status else None,
        "review_score": state.review_score,
        "needs_retry": state.needs_retry,
        "duration": state.workflow_metadata.duration,
        "error_count": len(state.error_messages),
        "debug_logs_count": len(state.debug_logs)
    }



