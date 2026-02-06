"""
MCP工具选择器

智能选择最合适的MCP工具来执行任务。
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import threading

from ..models import (
    TaskType, ToolCategory, MCPToolDefinition,
    MCPToolSelection, Task
)
from .registry import MCPToolRegistry


class ToolSelector:
    """智能工具选择器"""

    def __init__(self, registry: MCPToolRegistry):
        """
        初始化工具选择器

        Args:
            registry: 工具注册表
        """
        self.registry = registry
        self._lock = threading.RLock()

        # 关键词映射缓存
        self._keyword_cache: Dict[str, List[str]] = {}
        self._capability_cache: Dict[str, List[str]] = {}

        # 初始化关键词映射
        self._init_keyword_mappings()

    def _init_keyword_mappings(self):
        """初始化关键词映射"""
        # 中英文关键词映射到工具名称
        self.keyword_mappings = {
            # 数据分析相关
            "数据|分析|统计|报表|图表|趋势|data|analysis|statistics|report|chart|trend": [
                "data_analysis_tool"
            ],

            # 代码分析相关
            "代码|编程|审查|质量|bug|调试|复杂性|security|vulnerability|code|programming|review|quality|debug|complexity": [
                "code_analysis_tool"
            ],

            # 搜索相关
            "搜索|查找|查询|信息|网页|web|search|find|query|information|webpage": [
                "web_search_tool"
            ],

            # 文件系统相关
            "文件|目录|读取|写入|创建|删除|file|directory|read|write|create|delete": [
                "file_system_tool"
            ],

            # 网络相关
            "网络|API|请求|HTTP|调用|network|api|request|http|call": [
                "network_tool"
            ],

            # 创意相关
            "创意|生成|写作|故事|设计|creative|generate|write|story|design": [
                "creative_tool"
            ]
        }

        # 能力关键词映射
        self.capability_keywords = {
            "数据查询|查询数据|data query": ["data_query", "analysis"],
            "代码质量|质量检查|code quality": ["code_quality", "security_scan"],
            "网页搜索|信息检索|web search": ["web_search", "information_retrieval"],
            "文件操作|文件管理|file operation": ["file_read", "file_write"],
            "API调用|网络请求|api call": ["http_request", "api_call"],
            "内容生成|创意写作|content generation": ["content_generation", "creative_writing"]
        }

    def select_tools_for_task(self,
                             task_type: TaskType,
                             user_input: str,
                             max_tools: int = 3,
                             context: Optional[Dict[str, Any]] = None) -> List[MCPToolSelection]:
        """
        为任务选择合适的工具

        Args:
            task_type: 任务类型
            user_input: 用户输入
            max_tools: 最大工具数量
            context: 额外的上下文信息

        Returns:
            工具选择列表
        """
        candidates = []
        seen_tools = set()

        # 策略1: 基于任务类型
        task_tools = self.registry.get_tools_for_task(task_type)
        for tool in task_tools:
            if tool.name not in seen_tools:
                confidence = self._calculate_task_relevance(tool, task_type, user_input)
                candidates.append((tool, confidence))
                seen_tools.add(tool.name)

        # 策略2: 基于关键词匹配
        keyword_tools = self._select_by_keywords(user_input)
        for tool_name in keyword_tools:
            if tool_name not in seen_tools:
                tool = self.registry.get_tool(tool_name)
                if tool:
                    confidence = self._calculate_keyword_relevance(tool, user_input)
                    candidates.append((tool, confidence))
                    seen_tools.add(tool_name)

        # 策略3: 基于能力匹配
        capability_tools = self._select_by_capabilities(task_type, user_input)
        for tool_name in capability_tools:
            if tool_name not in seen_tools:
                tool = self.registry.get_tool(tool_name)
                if tool:
                    confidence = self._calculate_capability_relevance(tool, task_type, user_input)
                    candidates.append((tool, confidence))
                    seen_tools.add(tool_name)

        # 策略4: 基于上下文（如果提供）
        if context:
            context_tools = self._select_by_context(context)
            for tool_name in context_tools:
                if tool_name not in seen_tools:
                    tool = self.registry.get_tool(tool_name)
                    if tool:
                        confidence = self._calculate_context_relevance(tool, context)
                        candidates.append((tool, confidence))
                        seen_tools.add(tool_name)

        # 排序并选择最佳工具
        candidates.sort(key=lambda x: x[1], reverse=True)  # 按置信度降序

        selected_tools = []
        for tool, confidence in candidates[:max_tools]:
            selection = MCPToolSelection(
                tool_name=tool.name,
                reason=self._generate_selection_reason(tool, confidence, task_type, user_input),
                confidence=confidence,
                parameters=self._suggest_parameters(tool, task_type, user_input)
            )
            selected_tools.append(selection)

        return selected_tools

    def _calculate_task_relevance(self,
                                tool: MCPToolDefinition,
                                task_type: TaskType,
                                user_input: str) -> float:
        """
        计算工具与任务的相关性

        Args:
            tool: 工具定义
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            相关性分数 (0-1)
        """
        # 基础分数：任务类型匹配
        base_score = 0.8 if task_type in tool.applicable_tasks else 0.2

        # 优先级影响
        priority_bonus = (tool.priority - 1) / 9 * 0.2  # 0.1-0.2

        # 输入相关性
        input_relevance = self._calculate_input_relevance(tool, user_input)

        # 综合评分
        final_score = min(1.0, base_score + priority_bonus + input_relevance)

        return round(final_score, 3)

    def _calculate_keyword_relevance(self, tool: MCPToolDefinition, user_input: str) -> float:
        """
        计算关键词相关性

        Args:
            tool: 工具定义
            user_input: 用户输入

        Returns:
            相关性分数 (0-1)
        """
        input_lower = user_input.lower()
        total_keywords = 0
        matched_keywords = 0

        for keyword_pattern, tool_names in self.keyword_mappings.items():
            if tool.name in tool_names:
                keywords = keyword_pattern.split('|')
                total_keywords += len(keywords)

                for keyword in keywords:
                    if keyword.strip().lower() in input_lower:
                        matched_keywords += 1

        if total_keywords == 0:
            return 0.0

        # 关键词匹配度 + 基础分数
        keyword_score = matched_keywords / total_keywords
        base_score = 0.3  # 关键词匹配的基础分数

        return round(min(1.0, base_score + keyword_score * 0.7), 3)

    def _calculate_capability_relevance(self,
                                      tool: MCPToolDefinition,
                                      task_type: TaskType,
                                      user_input: str) -> float:
        """
        计算能力相关性

        Args:
            tool: 工具定义
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            相关性分数 (0-1)
        """
        input_lower = user_input.lower()
        capability_score = 0.0

        for capability in tool.capabilities:
            # 检查能力关键词
            for keyword_pattern, required_capabilities in self.capability_keywords.items():
                if capability in required_capabilities:
                    keywords = keyword_pattern.split('|')
                    for keyword in keywords:
                        if keyword.strip().lower() in input_lower:
                            capability_score += 0.2
                            break

        # 任务类型能力匹配
        task_capability_map = {
            TaskType.DATA_ANALYSIS: ["data_query", "analysis", "aggregation"],
            TaskType.CODE_REVIEW: ["code_quality", "security_scan", "complexity_analysis"],
            TaskType.RESEARCH: ["web_search", "information_retrieval"],
            TaskType.DOCUMENTATION: ["file_read", "file_write", "content_generation"],
            TaskType.CREATIVE_WRITING: ["content_generation", "creative_writing"],
            TaskType.BUG_ANALYSIS: ["code_quality", "debugging"],
            TaskType.PERFORMANCE_OPTIMIZATION: ["analysis", "performance_monitoring"]
        }

        required_capabilities = task_capability_map.get(task_type, [])
        capability_match = sum(1 for cap in required_capabilities if cap in tool.capabilities)

        task_score = capability_match / len(required_capabilities) if required_capabilities else 0.0

        final_score = min(1.0, capability_score + task_score * 0.6)
        return round(final_score, 3)

    def _calculate_context_relevance(self, tool: MCPToolDefinition, context: Dict[str, Any]) -> float:
        """
        计算上下文相关性

        Args:
            tool: 工具定义
            context: 上下文信息

        Returns:
            相关性分数 (0-1)
        """
        # 基于上下文中的历史工具使用情况
        previous_tools = context.get("previous_tools", [])
        if tool.name in previous_tools:
            return 0.9  # 历史使用过的工具高优先级

        # 基于上下文中的偏好设置
        preferred_categories = context.get("preferred_categories", [])
        if tool.category.value in preferred_categories:
            return 0.8

        # 基于上下文中的排除设置
        excluded_tools = context.get("excluded_tools", [])
        if tool.name in excluded_tools:
            return 0.0

        return 0.5  # 默认中等相关性

    def _calculate_input_relevance(self, tool: MCPToolDefinition, user_input: str) -> float:
        """
        计算输入相关性

        Args:
            tool: 工具定义
            user_input: 用户输入

        Returns:
            相关性分数 (0-1)
        """
        input_lower = user_input.lower()
        description_lower = tool.description.lower()

        # 简单文本匹配
        input_words = set(input_lower.split())
        desc_words = set(description_lower.split())

        # 计算词语重叠度
        overlap = len(input_words & desc_words)
        total_words = len(input_words | desc_words)

        if total_words == 0:
            return 0.0

        return round(overlap / total_words * 0.3, 3)  # 最高0.3的加成

    def _select_by_keywords(self, user_input: str) -> List[str]:
        """
        基于关键词选择工具

        Args:
            user_input: 用户输入

        Returns:
            工具名称列表
        """
        cache_key = hash(user_input)
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        input_lower = user_input.lower()
        matched_tools = set()

        for keyword_pattern, tool_names in self.keyword_mappings.items():
            if re.search(keyword_pattern, input_lower, re.IGNORECASE):
                matched_tools.update(tool_names)

        result = list(matched_tools)
        self._keyword_cache[cache_key] = result
        return result

    def _select_by_capabilities(self, task_type: TaskType, user_input: str) -> List[str]:
        """
        基于能力选择工具

        Args:
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            工具名称列表
        """
        cache_key = f"{task_type.value}:{hash(user_input)}"
        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key]

        input_lower = user_input.lower()
        matched_tools = set()

        # 获取任务相关的能力
        task_capabilities = self._get_task_capabilities(task_type)

        # 检查输入中提到的能力
        for keyword_pattern, capabilities in self.capability_keywords.items():
            if re.search(keyword_pattern, input_lower, re.IGNORECASE):
                # 找到匹配的能力，获取对应的工具
                for capability in capabilities:
                    if capability in task_capabilities:
                        tools = self.registry.get_tools_by_capability(capability)
                        matched_tools.update(tool.name for tool in tools)

        result = list(matched_tools)
        self._capability_cache[cache_key] = result
        return result

    def _select_by_context(self, context: Dict[str, Any]) -> List[str]:
        """
        基于上下文选择工具

        Args:
            context: 上下文信息

        Returns:
            工具名称列表
        """
        # 基于历史偏好
        preferred_tools = context.get("preferred_tools", [])
        if preferred_tools:
            return preferred_tools[:3]  # 最多返回3个

        # 基于类别偏好
        preferred_categories = context.get("preferred_categories", [])
        if preferred_categories:
            matched_tools = []
            for category_name in preferred_categories:
                try:
                    category = ToolCategory(category_name)
                    tools = self.registry.get_tools_by_category(category)
                    matched_tools.extend(tool.name for tool in tools[:2])  # 每个类别最多2个
                except ValueError:
                    continue
            return matched_tools[:3]

        return []

    def _get_task_capabilities(self, task_type: TaskType) -> List[str]:
        """
        获取任务类型对应的能力

        Args:
            task_type: 任务类型

        Returns:
            能力列表
        """
        capability_map = {
            TaskType.DATA_ANALYSIS: ["data_query", "analysis", "aggregation"],
            TaskType.CODE_REVIEW: ["code_quality", "security_scan", "complexity_analysis"],
            TaskType.RESEARCH: ["web_search", "information_retrieval"],
            TaskType.DOCUMENTATION: ["file_read", "file_write", "content_generation"],
            TaskType.CREATIVE_WRITING: ["content_generation", "creative_writing"],
            TaskType.BUG_ANALYSIS: ["code_quality", "debugging"],
            TaskType.PERFORMANCE_OPTIMIZATION: ["analysis", "performance_monitoring"],
            TaskType.GENERAL_CONSULTATION: ["web_search", "information_retrieval"]
        }

        return capability_map.get(task_type, [])

    def _generate_selection_reason(self,
                                 tool: MCPToolDefinition,
                                 confidence: float,
                                 task_type: TaskType,
                                 user_input: str) -> str:
        """
        生成工具选择理由

        Args:
            tool: 工具定义
            confidence: 置信度
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            选择理由
        """
        reasons = []

        # 任务类型匹配
        if task_type in tool.applicable_tasks:
            reasons.append(f"适用于{task_type.value}任务")

        # 关键词匹配
        input_lower = user_input.lower()
        for keyword_pattern, tool_names in self.keyword_mappings.items():
            if tool.name in tool_names and re.search(keyword_pattern, input_lower, re.IGNORECASE):
                reasons.append("关键词匹配")
                break

        # 能力匹配
        task_capabilities = self._get_task_capabilities(task_type)
        capability_matches = [cap for cap in tool.capabilities if cap in task_capabilities]
        if capability_matches:
            reasons.append(f"具备{', '.join(capability_matches)}能力")

        # 置信度描述
        if confidence > 0.8:
            reasons.append("高置信度匹配")
        elif confidence > 0.6:
            reasons.append("中等置信度匹配")
        else:
            reasons.append("基础匹配")

        return "、".join(reasons) if reasons else "通用工具选择"

    def _suggest_parameters(self,
                          tool: MCPToolDefinition,
                          task_type: TaskType,
                          user_input: str) -> Dict[str, Any]:
        """
        建议工具参数

        Args:
            tool: 工具定义
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            参数建议
        """
        suggested_params = {}

        # 基于任务类型设置默认参数
        if task_type == TaskType.DATA_ANALYSIS and tool.name == "data_analysis_tool":
            suggested_params = {
                "limit": 100,
                "time_range": {"days": 30}
            }

        elif task_type == TaskType.RESEARCH and tool.name == "web_search_tool":
            suggested_params = {
                "max_results": 10
            }

        elif task_type == TaskType.CREATIVE_WRITING and tool.name == "creative_tool":
            suggested_params = {
                "content_type": "article",
                "style": "professional",
                "length": "medium"
            }

        # 从用户输入中提取参数
        input_lower = user_input.lower()

        # 提取数字参数
        import re
        numbers = re.findall(r'\d+', input_lower)
        if numbers and tool.name == "data_analysis_tool":
            suggested_params["limit"] = min(int(numbers[0]), 1000)

        return suggested_params

    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        获取选择统计信息

        Returns:
            统计信息
        """
        with self._lock:
            return {
                "keyword_cache_size": len(self._keyword_cache),
                "capability_cache_size": len(self._capability_cache),
                "registry_tools_count": len(self.registry.tools)
            }



