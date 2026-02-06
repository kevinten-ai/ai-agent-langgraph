"""
MCP客户端 - Model Context Protocol HTTP客户端适配器

实现与MCP服务器的HTTP通信，支持连接池、缓存、重试等功能。
"""

import asyncio
import aiohttp
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin
import threading
from concurrent.futures import ThreadPoolExecutor

from ..models import MCPExecutionResult, ErrorInfo
from ..utils import StateManager


class ConnectionPool:
    """连接池管理器"""

    def __init__(self, max_connections: int = 10, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self._pools: Dict[str, aiohttp.ClientSession] = {}
        self._lock = threading.RLock()

    async def get_session(self, server_url: str) -> aiohttp.ClientSession:
        """获取会话"""
        with self._lock:
            if server_url not in self._pools:
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections
                )
                timeout = aiohttp.ClientTimeout(total=self.timeout)

                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'LangGraph-MCP-Client/1.0'
                    }
                )
                self._pools[server_url] = session

            return self._pools[server_url]

    async def close_all(self):
        """关闭所有连接"""
        with self._lock:
            close_tasks = []
            for session in self._pools.values():
                close_tasks.append(session.close())

            await asyncio.gather(*close_tasks, return_exceptions=True)
            self._pools.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._lock:
            return {
                "active_pools": len(self._pools),
                "total_connections": len(self._pools) * self.max_connections,
                "servers": list(self._pools.keys())
            }


class CacheManager:
    """缓存管理器"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _generate_key(self, server_url: str, tool_name: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 对参数进行标准化排序
        sorted_params = json.dumps(parameters, sort_keys=True)
        content = f"{server_url}:{tool_name}:{sorted_params}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get(self, server_url: str, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """获取缓存结果"""
        cache_key = self._generate_key(server_url, tool_name, parameters)

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not self._is_expired(entry):
                    return entry["data"]
                else:
                    # 过期删除
                    del self._cache[cache_key]

        return None

    async def set(self, server_url: str, tool_name: str, parameters: Dict[str, Any], data: Any):
        """设置缓存"""
        cache_key = self._generate_key(server_url, tool_name, parameters)

        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size:
                self._cleanup_expired()

            # 如果仍然超过限制，删除最旧的条目
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k]["timestamp"]
                )
                del self._cache[oldest_key]

            # 添加新条目
            self._cache[cache_key] = {
                "data": data,
                "timestamp": time.time(),
                "ttl": self.ttl_seconds
            }

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查条目是否过期"""
        return time.time() - entry["timestamp"] > entry["ttl"]

    def _cleanup_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if current_time - entry["timestamp"] > entry["ttl"]:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_entries = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if self._is_expired(entry))

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "hit_rate_estimate": 0.0,  # 需要额外的命中计数来计算
                "memory_usage_estimate": total_entries * 0.001  # 估算每条1KB
            }

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


class MCPClient:
    """MCP HTTP客户端适配器"""

    def __init__(self,
                 connection_pool: Optional[ConnectionPool] = None,
                 cache_manager: Optional[CacheManager] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 state_manager: Optional[StateManager] = None):
        """
        初始化MCP客户端

        Args:
            connection_pool: 连接池实例
            cache_manager: 缓存管理器实例
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            state_manager: 状态管理器实例
        """
        self.connection_pool = connection_pool or ConnectionPool()
        self.cache_manager = cache_manager or CacheManager()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.state_manager = state_manager or StateManager()

        # 客户端统计信息
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0
        }
        self._stats_lock = threading.RLock()

    async def call_tool(self,
                        server_url: str,
                        tool_name: str,
                        parameters: Dict[str, Any],
                        use_cache: bool = True) -> MCPExecutionResult:
        """
        调用MCP工具

        Args:
            server_url: MCP服务器URL
            tool_name: 工具名称
            parameters: 工具参数
            use_cache: 是否使用缓存

        Returns:
            执行结果
        """
        start_time = time.time()

        try:
            # 缓存检查
            if use_cache:
                cached_result = await self.cache_manager.get(server_url, tool_name, parameters)
                if cached_result is not None:
                    self._update_stats(cache_hit=True, response_time=time.time() - start_time)
                    return MCPExecutionResult(
                        success=True,
                        tool_name=tool_name,
                        result=cached_result,
                        execution_time=time.time() - start_time
                    )

            # 执行工具调用
            result = await self._call_tool_with_retry(server_url, tool_name, parameters)

            # 缓存结果
            if use_cache and result.success:
                await self.cache_manager.set(server_url, tool_name, parameters, result.result)

            # 更新统计信息
            self._update_stats(
                success=result.success,
                response_time=time.time() - start_time
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(success=False, response_time=execution_time)

            return MCPExecutionResult(
                success=False,
                tool_name=tool_name,
                error_message=str(e),
                execution_time=execution_time
            )

    async def _call_tool_with_retry(self,
                                   server_url: str,
                                   tool_name: str,
                                   parameters: Dict[str, Any]) -> MCPExecutionResult:
        """
        带重试的工具调用

        Args:
            server_url: 服务器URL
            tool_name: 工具名称
            parameters: 参数

        Returns:
            执行结果
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = await self._call_tool_once(server_url, tool_name, parameters)

                if result.success:
                    return result
                else:
                    # 如果是可重试的错误，继续重试
                    if self._is_retryable_error(result.error_message or ""):
                        last_exception = Exception(result.error_message)
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                    else:
                        return result

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue

        # 所有重试都失败
        return MCPExecutionResult(
            success=False,
            tool_name=tool_name,
            error_message=f"重试{self.max_retries}次后失败: {str(last_exception)}",
            execution_time=0.0
        )

    async def _call_tool_once(self,
                             server_url: str,
                             tool_name: str,
                             parameters: Dict[str, Any]) -> MCPExecutionResult:
        """
        单次工具调用

        Args:
            server_url: 服务器URL
            tool_name: 工具名称
            parameters: 参数

        Returns:
            执行结果
        """
        session = await self.connection_pool.get_session(server_url)

        # 构建请求数据
        request_data = {
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with session.post(
                urljoin(server_url, "/tool-endpoint"),
                json=request_data
            ) as response:

                if response.status == 200:
                    response_data = await response.json()

                    # 验证响应格式
                    if self._validate_response(response_data):
                        return MCPExecutionResult(
                            success=True,
                            tool_name=tool_name,
                            result=response_data.get("result"),
                            execution_time=float(response_data.get("execution_time", 0))
                        )
                    else:
                        return MCPExecutionResult(
                            success=False,
                            tool_name=tool_name,
                            error_message="无效的响应格式",
                            execution_time=0.0
                        )

                else:
                    error_text = await response.text()
                    return MCPExecutionResult(
                        success=False,
                        tool_name=tool_name,
                        error_message=f"HTTP {response.status}: {error_text}",
                        execution_time=0.0
                    )

        except asyncio.TimeoutError:
            return MCPExecutionResult(
                success=False,
                tool_name=tool_name,
                error_message="请求超时",
                execution_time=0.0
            )

        except aiohttp.ClientError as e:
            return MCPExecutionResult(
                success=False,
                tool_name=tool_name,
                error_message=f"网络错误: {str(e)}",
                execution_time=0.0
            )

        except Exception as e:
            return MCPExecutionResult(
                success=False,
                tool_name=tool_name,
                error_message=f"未知错误: {str(e)}",
                execution_time=0.0
            )

    def _validate_response(self, response_data: Dict[str, Any]) -> bool:
        """验证响应数据格式"""
        required_fields = ["success", "result"]
        return all(field in response_data for field in required_fields)

    def _is_retryable_error(self, error_message: str) -> bool:
        """判断错误是否可重试"""
        non_retryable_patterns = [
            "authentication failed",
            "authorization failed",
            "invalid parameters",
            "not found",
            "bad request"
        ]

        error_lower = error_message.lower()
        return not any(pattern in error_lower for pattern in non_retryable_patterns)

    def _update_stats(self, success: bool = False, cache_hit: bool = False, response_time: float = 0.0):
        """更新统计信息"""
        with self._stats_lock:
            self._stats["total_requests"] += 1

            if cache_hit:
                self._stats["cache_hits"] += 1
            elif success:
                self._stats["successful_requests"] += 1
            else:
                self._stats["failed_requests"] += 1

            # 更新平均响应时间
            if response_time > 0:
                current_avg = self._stats["average_response_time"]
                total_requests = self._stats["total_requests"]
                self._stats["average_response_time"] = (
                    (current_avg * (total_requests - 1)) + response_time
                ) / total_requests

    async def health_check(self, server_url: str) -> Dict[str, Any]:
        """健康检查"""
        try:
            session = await self.connection_pool.get_session(server_url)
            async with session.get(urljoin(server_url, "/health")) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "healthy", "details": data}
                else:
                    return {"status": "unhealthy", "status_code": response.status}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()

        # 添加连接池和缓存统计
        stats.update({
            "connection_pool": self.connection_pool.get_stats(),
            "cache": self.cache_manager.get_stats()
        })

        return stats

    async def close(self):
        """关闭客户端"""
        await self.connection_pool.close_all()
        self.cache_manager.clear()



