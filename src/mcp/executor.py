"""
MCP工具执行器

负责并行执行多个MCP工具调用，聚合结果并处理错误。
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from ..models import MCPToolSelection, MCPExecutionResult, ErrorInfo
from .client import MCPClient
from .registry import MCPToolRegistry


class MCPToolExecutor:
    """MCP工具执行器"""

    def __init__(self,
                 mcp_client: MCPClient,
                 registry: MCPToolRegistry,
                 max_concurrent: int = 5):
        """
        初始化执行器

        Args:
            mcp_client: MCP客户端
            registry: 工具注册表
            max_concurrent: 最大并发数
        """
        self.mcp_client = mcp_client
        self.registry = registry
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 统计信息
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
        self._stats_lock = threading.RLock()

    async def execute_parallel_primary(self,
                                     tool_selections: List[MCPToolSelection]) -> Dict[str, Any]:
        """
        并行执行主要工具

        Args:
            tool_selections: 工具选择列表

        Returns:
            执行结果汇总
        """
        if not tool_selections:
            return {
                "success": False,
                "error": "没有工具可执行",
                "primary_results": [],
                "failed_calls": [],
                "execution_summary": "无工具执行"
            }

        start_time = time.time()

        # 创建并发任务
        tasks = []
        for selection in tool_selections:
            task = self._execute_single_tool_with_semaphore(selection)
            tasks.append(task)

        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        primary_results = []
        failed_calls = []

        for i, result in enumerate(results):
            tool_name = tool_selections[i].tool_name

            if isinstance(result, Exception):
                failed_calls.append(tool_name)
                primary_results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0
                })
            else:
                primary_results.append(result)
                if not result.success:
                    failed_calls.append(tool_name)

        execution_time = time.time() - start_time
        execution_summary = f"执行完成: {len(primary_results)}/{len(tool_selections)} 成功，耗时 {execution_time:.2f}秒"

        # 更新统计信息
        self._update_stats(len(tool_selections), len(failed_calls), execution_time)

        return {
            "primary_results": primary_results,
            "failed_calls": failed_calls,
            "execution_summary": execution_summary,
            "total_execution_time": execution_time,
            "success_rate": (len(tool_selections) - len(failed_calls)) / len(tool_selections)
        }

    async def _execute_single_tool_with_semaphore(self,
                                                tool_selection: MCPToolSelection) -> MCPExecutionResult:
        """
        使用信号量控制并发执行单个工具

        Args:
            tool_selection: 工具选择

        Returns:
            执行结果
        """
        async with self.semaphore:
            return await self._execute_single_tool(tool_selection)

    async def _execute_single_tool(self, tool_selection: MCPToolSelection) -> MCPExecutionResult:
        """
        执行单个工具

        Args:
            tool_selection: 工具选择

        Returns:
            执行结果
        """
        tool_name = tool_selection.tool_name
        parameters = tool_selection.parameters or {}

        try:
            # 获取工具定义
            tool_def = self.registry.get_tool(tool_name)
            if not tool_def:
                return MCPExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    error_message=f"工具 '{tool_name}' 未找到",
                    execution_time=0.0
                )

            # 执行工具调用
            result = await self.mcp_client.call_tool(
                server_url=tool_def.server_url,
                tool_name=tool_def.tool_name,
                parameters=parameters,
                use_cache=tool_def.cache_enabled
            )

            return result

        except Exception as e:
            return MCPExecutionResult(
                success=False,
                tool_name=tool_name,
                error_message=f"执行异常: {str(e)}",
                execution_time=0.0
            )

    async def execute_with_fallback(self,
                                   primary_selections: List[MCPToolSelection],
                                   fallback_selections: Optional[List[MCPToolSelection]] = None) -> Dict[str, Any]:
        """
        执行工具并提供降级方案

        Args:
            primary_selections: 主要工具选择
            fallback_selections: 降级工具选择

        Returns:
            执行结果
        """
        # 先执行主要工具
        result = await self.execute_parallel_primary(primary_selections)

        # 如果主要工具全部失败，尝试降级方案
        if len(result["failed_calls"]) == len(primary_selections) and fallback_selections:
            print("主要工具全部失败，尝试降级方案...")
            fallback_result = await self.execute_parallel_primary(fallback_selections)

            result["fallback_attempted"] = True
            result["fallback_results"] = fallback_result
            result["execution_summary"] += f"；降级执行: {fallback_result['execution_summary']}"

        return result

    def _update_stats(self, total_tools: int, failed_count: int, execution_time: float):
        """更新统计信息"""
        with self._stats_lock:
            successful_count = total_tools - failed_count

            self._stats["total_executions"] += total_tools
            self._stats["successful_executions"] += successful_count
            self._stats["failed_executions"] += failed_count

            # 更新平均执行时间
            current_avg = self._stats["average_execution_time"]
            total_executions = self._stats["total_executions"]
            self._stats["average_execution_time"] = (
                (current_avg * (total_executions - total_tools)) + execution_time
            ) / total_executions

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()

            total = stats["total_executions"]
            if total > 0:
                stats["success_rate"] = stats["successful_executions"] / total
                stats["failure_rate"] = stats["failed_executions"] / total
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0

            return stats

    async def health_check_tools(self, tool_selections: List[MCPToolSelection]) -> Dict[str, Any]:
        """
        检查工具健康状态

        Args:
            tool_selections: 工具选择列表

        Returns:
            健康检查结果
        """
        health_results = {}

        for selection in tool_selections:
            tool_name = selection.tool_name
            tool_def = self.registry.get_tool(tool_name)

            if tool_def:
                # 检查服务器健康状态
                is_healthy = await self.mcp_client.health_check(tool_def.server_url)
                health_results[tool_name] = {
                    "healthy": is_healthy.get("status") == "healthy",
                    "server_url": tool_def.server_url,
                    "details": is_healthy
                }
            else:
                health_results[tool_name] = {
                    "healthy": False,
                    "error": "工具未注册"
                }

        healthy_count = sum(1 for result in health_results.values() if result["healthy"])

        return {
            "overall_healthy": healthy_count == len(tool_selections),
            "healthy_count": healthy_count,
            "total_count": len(tool_selections),
            "tool_health": health_results
        }



