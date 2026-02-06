"""
MCP工具注册表

管理所有注册的MCP工具，提供工具发现、注册、查询等功能。
"""

import json
import asyncio
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import threading
from datetime import datetime

from ..models import (
    MCPToolDefinition, MCPToolParameter, ToolCategory,
    TaskType, MCPToolSelection
)


class MCPToolRegistry:
    """MCP工具注册表"""

    def __init__(self):
        self.tools: Dict[str, MCPToolDefinition] = {}
        self._lock = threading.RLock()

        # 索引用于快速查询
        self._category_index: Dict[ToolCategory, Set[str]] = {}
        self._task_index: Dict[TaskType, Set[str]] = {}
        self._capability_index: Dict[str, Set[str]] = {}

        # 初始化内置工具
        self._initialize_builtin_tools()

    def _initialize_builtin_tools(self):
        """初始化内置工具定义"""
        builtin_tools = [
            # 数据分析工具
            MCPToolDefinition(
                name="data_analysis_tool",
                description="数据分析和查询工具，支持SQL查询和数据处理",
                category=ToolCategory.DATA_ANALYTICS,
                server_url="{data_analysis_server_url}",
                tool_name="analyze_data",
                parameters=[
                    MCPToolParameter(
                        name="query",
                        type="string",
                        description="查询条件或SQL语句",
                        required=True
                    ),
                    MCPToolParameter(
                        name="time_range",
                        type="object",
                        description="时间范围过滤",
                        required=False
                    ),
                    MCPToolParameter(
                        name="limit",
                        type="number",
                        description="结果限制数量",
                        required=False,
                        default=100
                    )
                ],
                capabilities=["data_query", "analysis", "aggregation", "filtering"],
                applicable_tasks=[
                    TaskType.DATA_ANALYSIS,
                    TaskType.PERFORMANCE_OPTIMIZATION,
                    TaskType.BUG_ANALYSIS
                ],
                priority=8,
                timeout_seconds=45,
                cache_enabled=True
            ),

            # 代码分析工具
            MCPToolDefinition(
                name="code_analysis_tool",
                description="代码质量分析和静态检查工具",
                category=ToolCategory.CODE_ANALYSIS,
                server_url="{code_analysis_server_url}",
                tool_name="analyze_code",
                parameters=[
                    MCPToolParameter(
                        name="code",
                        type="string",
                        description="待分析的代码内容",
                        required=True
                    ),
                    MCPToolParameter(
                        name="language",
                        type="string",
                        description="编程语言",
                        required=True,
                        enum=["python", "javascript", "java", "cpp", "go"]
                    ),
                    MCPToolParameter(
                        name="analysis_type",
                        type="string",
                        description="分析类型",
                        required=False,
                        enum=["complexity", "security", "performance", "style"],
                        default="complexity"
                    )
                ],
                capabilities=["code_quality", "security_scan", "complexity_analysis"],
                applicable_tasks=[
                    TaskType.CODE_REVIEW,
                    TaskType.BUG_ANALYSIS,
                    TaskType.PERFORMANCE_OPTIMIZATION
                ],
                priority=9,
                timeout_seconds=60,
                cache_enabled=False
            ),

            # 搜索工具
            MCPToolDefinition(
                name="web_search_tool",
                description="网页搜索和信息检索工具",
                category=ToolCategory.SEARCH,
                server_url="{search_server_url}",
                tool_name="web_search",
                parameters=[
                    MCPToolParameter(
                        name="query",
                        type="string",
                        description="搜索查询",
                        required=True
                    ),
                    MCPToolParameter(
                        name="max_results",
                        type="number",
                        description="最大结果数量",
                        required=False,
                        default=10
                    ),
                    MCPToolParameter(
                        name="include_domains",
                        type="array",
                        description="包含的域名列表",
                        required=False
                    ),
                    MCPToolParameter(
                        name="exclude_domains",
                        type="array",
                        description="排除的域名列表",
                        required=False
                    )
                ],
                capabilities=["web_search", "information_retrieval", "content_filtering"],
                applicable_tasks=[
                    TaskType.RESEARCH,
                    TaskType.GENERAL_CONSULTATION,
                    TaskType.DATA_ANALYSIS
                ],
                priority=7,
                timeout_seconds=30,
                cache_enabled=True
            ),

            # 文件系统工具
            MCPToolDefinition(
                name="file_system_tool",
                description="文件系统操作工具",
                category=ToolCategory.FILE_SYSTEM,
                server_url="{file_system_server_url}",
                tool_name="file_operation",
                parameters=[
                    MCPToolParameter(
                        name="operation",
                        type="string",
                        description="操作类型",
                        required=True,
                        enum=["read", "write", "list", "delete", "move"]
                    ),
                    MCPToolParameter(
                        name="path",
                        type="string",
                        description="文件或目录路径",
                        required=True
                    ),
                    MCPToolParameter(
                        name="content",
                        type="string",
                        description="写入内容（write操作时必需）",
                        required=False
                    ),
                    MCPToolParameter(
                        name="encoding",
                        type="string",
                        description="文件编码",
                        required=False,
                        default="utf-8"
                    )
                ],
                capabilities=["file_read", "file_write", "directory_list", "file_management"],
                applicable_tasks=[
                    TaskType.DOCUMENTATION,
                    TaskType.DATA_ANALYSIS,
                    TaskType.GENERAL_CONSULTATION
                ],
                priority=6,
                timeout_seconds=20,
                cache_enabled=False
            ),

            # 网络工具
            MCPToolDefinition(
                name="network_tool",
                description="网络请求和API调用工具",
                category=ToolCategory.NETWORKING,
                server_url="{network_server_url}",
                tool_name="http_request",
                parameters=[
                    MCPToolParameter(
                        name="method",
                        type="string",
                        description="HTTP方法",
                        required=True,
                        enum=["GET", "POST", "PUT", "DELETE", "PATCH"]
                    ),
                    MCPToolParameter(
                        name="url",
                        type="string",
                        description="请求URL",
                        required=True
                    ),
                    MCPToolParameter(
                        name="headers",
                        type="object",
                        description="请求头",
                        required=False
                    ),
                    MCPToolParameter(
                        name="body",
                        type="string",
                        description="请求体",
                        required=False
                    ),
                    MCPToolParameter(
                        name="timeout",
                        type="number",
                        description="超时时间（秒）",
                        required=False,
                        default=30
                    )
                ],
                capabilities=["http_request", "api_call", "web_scraping"],
                applicable_tasks=[
                    TaskType.RESEARCH,
                    TaskType.DATA_ANALYSIS,
                    TaskType.GENERAL_CONSULTATION
                ],
                priority=7,
                timeout_seconds=35,
                cache_enabled=True
            ),

            # 创意工具
            MCPToolDefinition(
                name="creative_tool",
                description="创意内容生成工具",
                category=ToolCategory.CREATIVE,
                server_url="{creative_server_url}",
                tool_name="generate_content",
                parameters=[
                    MCPToolParameter(
                        name="prompt",
                        type="string",
                        description="生成提示",
                        required=True
                    ),
                    MCPToolParameter(
                        name="content_type",
                        type="string",
                        description="内容类型",
                        required=False,
                        enum=["story", "poem", "article", "code", "design"],
                        default="article"
                    ),
                    MCPToolParameter(
                        name="style",
                        type="string",
                        description="生成风格",
                        required=False,
                        enum=["professional", "casual", "creative", "technical"],
                        default="professional"
                    ),
                    MCPToolParameter(
                        name="length",
                        type="string",
                        description="内容长度",
                        required=False,
                        enum=["short", "medium", "long"],
                        default="medium"
                    )
                ],
                capabilities=["content_generation", "creative_writing", "idea_generation"],
                applicable_tasks=[
                    TaskType.CREATIVE_WRITING,
                    TaskType.DOCUMENTATION,
                    TaskType.GENERAL_CONSULTATION
                ],
                priority=8,
                timeout_seconds=40,
                cache_enabled=False
            )
        ]

        for tool in builtin_tools:
            self.register_tool(tool)

    def register_tool(self, tool_definition: MCPToolDefinition):
        """
        注册工具

        Args:
            tool_definition: 工具定义
        """
        with self._lock:
            self.tools[tool_definition.name] = tool_definition

            # 更新索引
            self._update_indexes(tool_definition)

    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功注销
        """
        with self._lock:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                del self.tools[tool_name]

                # 更新索引
                self._remove_from_indexes(tool)

                return True

        return False

    def get_tool(self, tool_name: str) -> Optional[MCPToolDefinition]:
        """
        获取工具定义

        Args:
            tool_name: 工具名称

        Returns:
            工具定义，如果不存在则返回None
        """
        with self._lock:
            return self.tools.get(tool_name)

    def list_tools(self,
                  category: Optional[ToolCategory] = None,
                  applicable_task: Optional[TaskType] = None) -> List[MCPToolDefinition]:
        """
        列出工具

        Args:
            category: 工具类别过滤
            applicable_task: 适用的任务类型过滤

        Returns:
            工具列表
        """
        with self._lock:
            tools = list(self.tools.values())

            if category:
                tools = [t for t in tools if t.category == category]

            if applicable_task:
                tools = [t for t in tools if applicable_task in t.applicable_tasks]

            return tools

    def get_tools_for_task(self, task_type: TaskType) -> List[MCPToolDefinition]:
        """
        获取适用于特定任务的工具

        Args:
            task_type: 任务类型

        Returns:
            工具列表，按优先级排序
        """
        with self._lock:
            tool_names = self._task_index.get(task_type, set())
            tools = [self.tools[name] for name in tool_names if name in self.tools]

            # 按优先级降序排序
            return sorted(tools, key=lambda t: t.priority, reverse=True)

    def get_tools_by_category(self, category: ToolCategory) -> List[MCPToolDefinition]:
        """
        获取指定类别的工具

        Args:
            category: 工具类别

        Returns:
            工具列表
        """
        with self._lock:
            tool_names = self._category_index.get(category, set())
            return [self.tools[name] for name in tool_names if name in self.tools]

    def get_tools_by_capability(self, capability: str) -> List[MCPToolDefinition]:
        """
        获取具有指定能力的工具

        Args:
            capability: 能力名称

        Returns:
            工具列表
        """
        with self._lock:
            tool_names = self._capability_index.get(capability, set())
            tools = [self.tools[name] for name in tool_names if name in self.tools]

            # 按优先级排序
            return sorted(tools, key=lambda t: t.priority, reverse=True)

    def update_tool_config(self, tool_name: str, config_updates: Dict[str, Any]) -> bool:
        """
        更新工具配置

        Args:
            tool_name: 工具名称
            config_updates: 配置更新

        Returns:
            是否成功更新
        """
        with self._lock:
            if tool_name not in self.tools:
                return False

            tool = self.tools[tool_name]

            # 更新允许的字段
            updatable_fields = [
                'description', 'server_url', 'timeout_seconds',
                'cache_enabled', 'priority'
            ]

            for field in updatable_fields:
                if field in config_updates:
                    setattr(tool, field, config_updates[field])

            # 更新时间戳
            tool.updated_at = datetime.now()

            return True

    def _update_indexes(self, tool: MCPToolDefinition):
        """更新索引"""
        # 类别索引
        if tool.category not in self._category_index:
            self._category_index[tool.category] = set()
        self._category_index[tool.category].add(tool.name)

        # 任务索引
        for task_type in tool.applicable_tasks:
            if task_type not in self._task_index:
                self._task_index[task_type] = set()
            self._task_index[task_type].add(tool.name)

        # 能力索引
        for capability in tool.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(tool.name)

    def _remove_from_indexes(self, tool: MCPToolDefinition):
        """从索引中移除"""
        # 类别索引
        if tool.category in self._category_index:
            self._category_index[tool.category].discard(tool.name)

        # 任务索引
        for task_type in tool.applicable_tasks:
            if task_type in self._task_index:
                self._task_index[task_type].discard(tool.name)

        # 能力索引
        for capability in tool.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability].discard(tool.name)

    def save_to_file(self, file_path: str) -> bool:
        """
        保存注册表到文件

        Args:
            file_path: 文件路径

        Returns:
            是否成功保存
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                data = {
                    "tools": [tool.model_dump() for tool in self.tools.values()],
                    "exported_at": datetime.now().isoformat()
                }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"保存注册表失败: {e}")
            return False

    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载注册表

        Args:
            file_path: 文件路径

        Returns:
            是否成功加载
        """
        try:
            if not Path(file_path).exists():
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tools_data = data.get("tools", [])
            for tool_data in tools_data:
                tool = MCPToolDefinition(**tool_data)
                self.register_tool(tool)

            return True

        except Exception as e:
            print(f"加载注册表失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        with self._lock:
            total_tools = len(self.tools)
            category_stats = {}
            task_stats = {}

            for tool in self.tools.values():
                # 类别统计
                cat = tool.category.value
                category_stats[cat] = category_stats.get(cat, 0) + 1

                # 任务统计
                for task in tool.applicable_tasks:
                    task_name = task.value
                    task_stats[task_name] = task_stats.get(task_name, 0) + 1

            return {
                "total_tools": total_tools,
                "categories": category_stats,
                "task_coverage": task_stats,
                "average_priority": sum(t.priority for t in self.tools.values()) / total_tools if total_tools > 0 else 0
            }



