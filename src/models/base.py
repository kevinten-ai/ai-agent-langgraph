"""
基础数据模型定义

定义系统核心的数据类型和基础模型。
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """任务类型枚举"""
    GENERAL_CONSULTATION = "general_consultation"
    DATA_ANALYSIS = "data_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_ANALYSIS = "bug_analysis"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    CREATIVE_WRITING = "creative_writing"


class TaskPriority(str, Enum):
    """任务优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ToolCategory(str, Enum):
    """工具类别枚举"""
    DATA_ANALYTICS = "data_analytics"
    SEARCH = "search"
    CODE_ANALYSIS = "code_analysis"
    NETWORKING = "networking"
    FILE_SYSTEM = "file_system"
    COMMUNICATION = "communication"
    CREATIVE = "creative"


class BaseModelWithTimestamp(BaseModel):
    """带时间戳的基础模型"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()


class Task(BaseModelWithTimestamp):
    """任务模型"""

    id: str = Field(..., description="任务唯一标识")
    user_input: str = Field(..., description="用户原始输入")
    task_type: TaskType = Field(..., description="任务类型")
    task_description: str = Field("", description="任务描述")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="任务优先级")
    complexity: Literal["simple", "medium", "complex"] = Field("medium", description="任务复杂度")
    requirements: List[str] = Field(default_factory=list, description="任务要求")
    expected_outcome: str = Field("", description="期望结果")
    max_retries: int = Field(3, description="最大重试次数")
    timeout_seconds: int = Field(300, description="超时时间(秒)")

    # 元数据
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class MCPToolParameter(BaseModel):
    """MCP工具参数定义"""

    name: str = Field(..., description="参数名称")
    type: str = Field(..., description="参数类型")  # string, number, boolean, object, array
    description: str = Field("", description="参数描述")
    required: bool = Field(False, description="是否必需")
    default: Any = Field(None, description="默认值")
    enum: Optional[List[Any]] = Field(None, description="枚举值")


class MCPToolDefinition(BaseModel):
    """MCP工具定义"""

    name: str = Field(..., description="工具名称")
    description: str = Field("", description="工具描述")
    category: ToolCategory = Field(..., description="工具类别")
    server_url: str = Field(..., description="MCP服务器URL")
    tool_name: str = Field(..., description="服务器端工具名称")
    parameters: List[MCPToolParameter] = Field(default_factory=list, description="工具参数")
    capabilities: List[str] = Field(default_factory=list, description="工具能力")
    applicable_tasks: List[TaskType] = Field(default_factory=list, description="适用的任务类型")
    priority: int = Field(5, description="工具优先级 (1-10)", ge=1, le=10)
    timeout_seconds: int = Field(30, description="超时时间(秒)")
    cache_enabled: bool = Field(True, description="是否启用缓存")
    rate_limit: Optional[int] = Field(None, description="每分钟请求限制")

    # 元数据
    version: str = Field("1.0.0", description="工具版本")
    author: str = Field("", description="工具作者")
    documentation_url: Optional[str] = Field(None, description="文档URL")


class MCPToolSelection(BaseModel):
    """MCP工具选择结果"""

    tool_name: str = Field(..., description="选择的工具名称")
    reason: str = Field("", description="选择理由")
    confidence: float = Field(1.0, description="选择置信度")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="预设参数")


class MCPExecutionResult(BaseModel):
    """MCP执行结果"""

    success: bool = Field(..., description="是否成功")
    tool_name: str = Field("", description="执行的工具名称")
    result: Any = Field(None, description="执行结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(0.0, description="执行时间(秒)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="执行元数据")


class ErrorInfo(BaseModel):
    """错误信息"""

    error_type: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[str] = Field(None, description="详细错误信息")
    retryable: bool = Field(False, description="是否可以重试")
    suggested_action: Optional[str] = Field(None, description="建议行动")


class ExecutionResult(BaseModel):
    """执行结果"""

    content: str = Field("", description="执行内容")
    status: ExecutionStatus = Field(ExecutionStatus.PENDING, description="执行状态")
    confidence_score: float = Field(0.0, description="置信度分数")
    sources: List[str] = Field(default_factory=list, description="信息来源")
    execution_time: float = Field(0.0, description="执行时间")
    error_info: Optional[ErrorInfo] = Field(None, description="错误信息")


class ReviewResult(BaseModel):
    """审核结果"""

    overall_score: float = Field(0.0, description="总体评分")
    quality_score: float = Field(0.0, description="质量评分")
    consistency_score: float = Field(0.0, description="一致性评分")
    needs_retry: bool = Field(False, description="是否需要重试")
    feedback: str = Field("", description="审核反馈")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


class TaskAnalysisResult(BaseModel):
    """任务分析结果"""

    task_type: TaskType = Field(..., description="确定的任务类型")
    priority: TaskPriority = Field(..., description="确定的优先级")
    complexity: Literal["simple", "medium", "complex"] = Field("medium", description="确定的复杂度")
    recommended_tools: List[str] = Field(default_factory=list, description="推荐工具")
    reasoning: str = Field("", description="分析推理过程")
    estimated_duration: int = Field(60, description="预估耗时(秒)")
    required_capabilities: List[str] = Field(default_factory=list, description="所需能力")


class ErrorHandlingResult(BaseModel):
    """错误处理结果"""

    strategy: str = Field(..., description="处理策略")
    message: str = Field(..., description="处理消息")
    max_retries: int = Field(0, description="最大重试次数")
    error_details: str = Field("", description="错误详情")
    should_retry: bool = Field(False, description="是否应该重试")
    alternative_solutions: List[str] = Field(default_factory=list, description="替代方案")


class MCPErrorHandlingResult(BaseModel):
    """MCP错误处理结果"""

    needs_fallback: bool = Field(False, description="是否需要降级")
    alternative_tools: List[str] = Field(default_factory=list, description="替代工具")
    error_message: str = Field("", description="错误消息")
    fallback_reason: str = Field("", description="降级原因")


class WorkflowMetadata(BaseModel):
    """工作流元数据"""

    workflow_id: str = Field(..., description="工作流ID")
    workflow_version: str = Field("1.0.0", description="工作流版本")
    start_time: datetime = Field(default_factory=datetime.now, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration: float = Field(0.0, description="执行时长(秒)")
    agent_count: int = Field(0, description="参与Agent数量")
    tool_calls: int = Field(0, description="工具调用次数")
    retry_count: int = Field(0, description="重试次数")


class PerformanceMetrics(BaseModel):
    """性能指标"""

    execution_time: float = Field(0.0, description="执行时间")
    memory_usage: float = Field(0.0, description="内存使用")
    cpu_usage: float = Field(0.0, description="CPU使用率")
    network_requests: int = Field(0, description="网络请求数")
    cache_hits: int = Field(0, description="缓存命中数")
    error_count: int = Field(0, description="错误数量")



