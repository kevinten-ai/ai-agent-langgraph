"""
多Agent工作流编排器

整合所有Agent、MCP工具和状态管理，提供完整的工作流执行能力。
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from ..models import (
    AgentState, Task, TaskType, ExecutionStatus,
    create_initial_state, update_state_with_task_analysis,
    update_state_with_tool_selection, update_state_with_execution_result,
    update_state_with_review_result, finalize_state
)
from ..agents import TaskAssigner
from ..utils import StateManager
from ..mcp import MCPClient, MCPToolRegistry, ToolSelector, MCPToolExecutor


class MultiAgentWorkflow:
    """多Agent协作工作流"""

    def __init__(self,
                 enable_mcp_integration: bool = True,
                 llm: Optional[ChatOpenAI] = None,
                 state_manager: Optional[StateManager] = None):
        """
        初始化工作流

        Args:
            enable_mcp_integration: 是否启用MCP集成
            llm: 语言模型实例
            state_manager: 状态管理器实例
        """
        self.enable_mcp_integration = enable_mcp_integration
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.state_manager = state_manager or StateManager()

        # 初始化组件
        self.task_assigner = TaskAssigner(llm=self.llm, state_manager=self.state_manager)

        # MCP相关组件
        if enable_mcp_integration:
            self.mcp_client = MCPClient()
            self.tool_registry = MCPToolRegistry()
            self.tool_selector = ToolSelector(self.tool_registry)
            self.tool_executor = MCPToolExecutor(
                mcp_client=self.mcp_client,
                registry=self.tool_registry
            )

        # 构建工作流图
        self.workflow = StateGraph(AgentState)
        self._build_workflow()

        # 编译工作流
        self.app = self.workflow.compile()

    def _build_workflow(self):
        """构建工作流图"""
        # 添加节点
        self.workflow.add_node("task_assigner", self._task_assigner_node)
        self.workflow.add_node("tool_selector", self._tool_selector_node)
        self.workflow.add_node("mcp_executor", self._mcp_executor_node)
        self.workflow.add_node("executor", self._executor_node)
        self.workflow.add_node("reviewer", self._reviewer_node)

        # 设置入口
        self.workflow.set_entry_point("task_assigner")

        # 添加边
        self.workflow.add_edge("task_assigner", "tool_selector")

        if self.enable_mcp_integration:
            # MCP路径
            self.workflow.add_conditional_edges(
                "tool_selector",
                self._should_use_mcp,
                {
                    "mcp": "mcp_executor",
                    "direct": "executor"
                }
            )
            self.workflow.add_edge("mcp_executor", "executor")
        else:
            # 直接路径
            self.workflow.add_edge("tool_selector", "executor")

        # 审核和重试逻辑
        self.workflow.add_edge("executor", "reviewer")
        self.workflow.add_conditional_edges(
            "reviewer",
            self._should_retry,
            {
                "retry": "executor",
                "end": END
            }
        )

    async def run_workflow(self, user_input: str, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整工作流

        Args:
            user_input: 用户输入
            workflow_id: 工作流ID（可选）

        Returns:
            工作流执行结果
        """
        # 创建初始状态
        initial_state = create_initial_state(user_input)
        if workflow_id:
            initial_state.workflow_metadata.workflow_id = workflow_id

        try:
            # 执行工作流
            result = await self.app.ainvoke(initial_state)

            # 最终化状态
            final_state = finalize_state(result, result.final_answer or "工作流完成")

            # 保存到状态管理器
            if self.state_manager:
                self.state_manager.update_state(
                    final_state.workflow_metadata.workflow_id,
                    final_state.model_dump()
                )

            return {
                "success": True,
                "workflow_id": final_state.workflow_metadata.workflow_id,
                "final_answer": final_state.final_answer,
                "execution_summary": self._generate_execution_summary(final_state),
                "metadata": final_state.workflow_metadata.model_dump()
            }

        except Exception as e:
            # 错误处理
            error_state = create_initial_state(user_input)
            error_state.workflow_status = "failed"
            error_state.error_messages = [f"工作流执行失败: {str(e)}"]

            return {
                "success": False,
                "error": str(e),
                "workflow_id": error_state.workflow_metadata.workflow_id
            }

    async def _task_assigner_node(self, state: AgentState) -> AgentState:
        """任务分配节点"""
        try:
            updated_state = await self.task_assigner.assign_task(state)

            # 添加调试日志
            updated_state.debug_logs.append(
                f"任务分配完成: {updated_state.task_type} - {updated_state.task_description[:50]}..."
            )

            return updated_state

        except Exception as e:
            state.error_messages.append(f"任务分配失败: {str(e)}")
            state.workflow_status = "failed"
            return state

    async def _tool_selector_node(self, state: AgentState) -> AgentState:
        """工具选择节点"""
        if not self.enable_mcp_integration or not state.task:
            # 不使用MCP或没有任务，直接返回
            state.selected_tools = []
            return state

        try:
            # 使用工具选择器
            tool_selections = self.tool_selector.select_tools_for_task(
                task_type=state.task_type,
                user_input=state.user_input,
                max_tools=3
            )

            # 更新状态
            updated_state = update_state_with_tool_selection(state, tool_selections)
            updated_state.debug_logs.append(f"选择工具: {[t.tool_name for t in tool_selections]}")

            return updated_state

        except Exception as e:
            state.error_messages.append(f"工具选择失败: {str(e)}")
            state.selected_tools = []
            return state

    async def _mcp_executor_node(self, state: AgentState) -> AgentState:
        """MCP执行节点"""
        if not self.enable_mcp_integration or not state.selected_tools:
            return state

        try:
            # 执行MCP工具
            execution_result = await self.tool_executor.execute_parallel_primary(state.selected_tools)

            # 更新状态
            state.mcp_results = execution_result["primary_results"]
            state.debug_logs.append(f"MCP执行完成: {execution_result['execution_summary']}")

            # 检查是否有失败的调用
            if execution_result["failed_calls"]:
                state.debug_logs.append(f"失败的工具调用: {execution_result['failed_calls']}")

            return state

        except Exception as e:
            state.error_messages.append(f"MCP执行失败: {str(e)}")
            return state

    async def _executor_node(self, state: AgentState) -> AgentState:
        """执行节点 - 整合MCP结果并生成最终回答"""
        try:
            # 使用LLM整合所有信息生成回答
            final_answer = await self._generate_final_answer(state)

            # 创建执行结果
            from ..models import ExecutionResult
            execution_result = ExecutionResult(
                content=final_answer,
                status=ExecutionStatus.SUCCESS,
                confidence_score=0.8,  # 简化为固定值
                sources=self._extract_sources(state),
                execution_time=state.workflow_metadata.duration
            )

            # 更新状态
            updated_state = update_state_with_execution_result(state, execution_result)
            updated_state.final_answer = final_answer

            return updated_state

        except Exception as e:
            state.error_messages.append(f"执行失败: {str(e)}")
            state.execution_status = ExecutionStatus.FAILED
            return state

    async def _reviewer_node(self, state: AgentState) -> AgentState:
        """审核节点"""
        try:
            # 简化的审核逻辑
            review_score = self._calculate_review_score(state)

            # 决定是否需要重试
            needs_retry = review_score < 6.0 and state.retry_count < 2

            if needs_retry:
                state.retry_count += 1
                state.needs_retry = True
                state.debug_logs.append(f"审核不通过 (分数: {review_score})，准备重试")
            else:
                state.needs_retry = False
                state.debug_logs.append(f"审核通过 (分数: {review_score})")

            # 更新审核结果
            from ..models import ReviewResult
            review_result = ReviewResult(
                overall_score=review_score,
                quality_score=review_score,
                consistency_score=review_score,
                needs_retry=needs_retry,
                feedback=self._generate_review_feedback(review_score)
            )

            updated_state = update_state_with_review_result(state, review_result)

            return updated_state

        except Exception as e:
            state.error_messages.append(f"审核失败: {str(e)}")
            state.needs_retry = False
            return state

    async def _generate_final_answer(self, state: AgentState) -> str:
        """生成最终答案"""
        # 构建上下文
        context_parts = []

        # 用户输入
        context_parts.append(f"用户查询: {state.user_input}")

        # 任务信息
        if state.task:
            context_parts.append(f"任务类型: {state.task_type.value}")
            context_parts.append(f"任务描述: {state.task_description}")

        # MCP结果
        if state.mcp_results:
            mcp_content = []
            for result in state.mcp_results:
                if result.get("success") and result.get("result"):
                    mcp_content.append(f"- {result['tool_name']}: {str(result['result'])[:200]}...")
            if mcp_content:
                context_parts.append(f"MCP工具结果:\n" + "\n".join(mcp_content))

        context = "\n\n".join(context_parts)

        # 生成最终答案
        prompt = f"""基于以下信息，为用户查询提供完整的回答：

{context}

请提供：
1. 清晰的答案
2. 必要的解释
3. 相关的数据支持
4. 如果适用的话，提供下一步建议

回答要专业、准确、有帮助。"""

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content

    def _extract_sources(self, state: AgentState) -> List[str]:
        """提取信息来源"""
        sources = []

        # MCP工具作为来源
        if state.mcp_results:
            for result in state.mcp_results:
                if result.get("success"):
                    sources.append(f"MCP工具: {result['tool_name']}")

        return sources

    def _calculate_review_score(self, state: AgentState) -> float:
        """计算审核分数"""
        score = 7.0  # 基础分数

        # 基于执行状态调整
        if state.execution_status == ExecutionStatus.SUCCESS:
            score += 2.0
        elif state.execution_status == ExecutionStatus.PARTIAL:
            score += 0.5
        else:
            score -= 2.0

        # 基于错误数量调整
        error_count = len(state.error_messages)
        score -= min(error_count * 0.5, 2.0)

        # 基于重试次数调整
        retry_penalty = state.retry_count * 0.3
        score -= min(retry_penalty, 1.0)

        return max(0.0, min(10.0, score))

    def _generate_review_feedback(self, score: float) -> str:
        """生成审核反馈"""
        if score >= 8.0:
            return "回答质量优秀，逻辑清晰，信息准确。"
        elif score >= 6.0:
            return "回答质量良好，但可以进一步改进。"
        else:
            return "回答质量需要改进，建议重新生成。"

    def _should_use_mcp(self, state: AgentState) -> str:
        """判断是否使用MCP"""
        return "mcp" if state.selected_tools else "direct"

    def _should_retry(self, state: AgentState) -> str:
        """判断是否重试"""
        return "retry" if state.needs_retry else "end"

    def _generate_execution_summary(self, state: AgentState) -> Dict[str, Any]:
        """生成执行摘要"""
        return {
            "task_type": state.task_type.value if state.task_type else None,
            "execution_status": state.execution_status.value if state.execution_status else None,
            "review_score": state.review_score,
            "retry_count": state.retry_count,
            "tool_calls": len(state.selected_tools) if state.selected_tools else 0,
            "mcp_results_count": len(state.mcp_results) if state.mcp_results else 0,
            "error_count": len(state.error_messages),
            "execution_time": state.workflow_metadata.duration
        }

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        state = self.state_manager.get_state(workflow_id)
        if state:
            return {
                "workflow_id": workflow_id,
                "status": state.workflow_status,
                "current_agent": state.current_agent,
                "progress": self._calculate_progress(state),
                "last_update": state.workflow_metadata.end_time or state.workflow_metadata.start_time
            }
        return None

    def _calculate_progress(self, state: AgentState) -> float:
        """计算执行进度"""
        steps_completed = 0
        total_steps = 4  # task_assigner, tool_selector, executor, reviewer

        if state.task_type:
            steps_completed += 1
        if state.selected_tools is not None:
            steps_completed += 1
        if state.execution_result:
            steps_completed += 1
        if state.review_score is not None:
            steps_completed += 1

        return steps_completed / total_steps

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        state = self.state_manager.get_state(workflow_id)
        if state and state.workflow_status not in ["completed", "failed"]:
            state.workflow_status = "cancelled"
            state.workflow_metadata.end_time = datetime.now()
            self.state_manager.update_state(workflow_id, {"workflow_status": "cancelled"})
            return True
        return False



