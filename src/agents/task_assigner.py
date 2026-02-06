"""
TaskAssigner Agent - 任务分配Agent

负责分析用户输入，确定任务类型、优先级和复杂度，并推荐合适的工具。
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..models import (
    Task, TaskType, TaskPriority, TaskAnalysisResult,
    AgentState, update_state_with_task_analysis
)
from ..utils import StateManager


class TaskAssigner:
    """任务分配Agent"""

    def __init__(self,
                 llm: Optional[ChatOpenAI] = None,
                 state_manager: Optional[StateManager] = None):
        """
        初始化TaskAssigner

        Args:
            llm: 语言模型实例
            state_manager: 状态管理器实例
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.state_manager = state_manager or StateManager()

        # 初始化关键词映射
        self._init_keyword_mappings()

        # 初始化复杂度评估规则
        self._init_complexity_rules()

    def _init_keyword_mappings(self):
        """初始化关键词映射"""
        self.keyword_mappings = {
            TaskType.DATA_ANALYSIS: [
                '分析', '统计', '数据', '报表', '图表', '趋势',
                'analyze', 'statistics', 'data', 'report', 'chart', 'trend'
            ],
            TaskType.PERFORMANCE_OPTIMIZATION: [
                '优化', '性能', '效率', '速度', '内存', 'CPU',
                'optimize', 'performance', 'efficiency', 'speed', 'memory', 'cpu'
            ],
            TaskType.BUG_ANALYSIS: [
                '错误', 'bug', '异常', '调试', '修复', '问题',
                'error', 'bug', 'exception', 'debug', 'fix', 'issue'
            ],
            TaskType.CODE_REVIEW: [
                '审查', 'review', '检查', '评估', '质量', '标准',
                'review', 'check', 'evaluate', 'quality', 'standard'
            ],
            TaskType.DOCUMENTATION: [
                '文档', '说明', '指南', '教程', '记录',
                'document', 'guide', 'tutorial', 'record'
            ],
            TaskType.RESEARCH: [
                '研究', '调查', '探索', '发现', '学习',
                'research', 'investigate', 'explore', 'discover', 'learn'
            ],
            TaskType.CREATIVE_WRITING: [
                '创作', '写作', '设计', '生成', '创意',
                'create', 'write', 'design', 'generate', 'creative'
            ],
            TaskType.GENERAL_CONSULTATION: [
                '咨询', '建议', '帮助', '解释', '了解',
                'consult', 'advice', 'help', 'explain', 'understand'
            ]
        }

    def _init_complexity_rules(self):
        """初始化复杂度评估规则"""
        self.complexity_rules = {
            'length_threshold': 200,  # 字符长度阈值
            'keyword_complexity': {
                '非常': 2, '极其': 2, '复杂': 1, '困难': 1,
                '深度': 1, '详细': 1, '全面': 1, '系统性': 2,
                '多维度': 1, '跨领域': 1, '创新性': 1
            },
            'task_type_complexity': {
                TaskType.DATA_ANALYSIS: 1.5,
                TaskType.PERFORMANCE_OPTIMIZATION: 2.0,
                TaskType.BUG_ANALYSIS: 1.8,
                TaskType.RESEARCH: 1.7,
                TaskType.CODE_REVIEW: 1.3,
                TaskType.CREATIVE_WRITING: 1.2,
                TaskType.DOCUMENTATION: 1.0,
                TaskType.GENERAL_CONSULTATION: 1.0
            }
        }

    async def analyze_task(self, user_input: str) -> TaskAnalysisResult:
        """
        分析用户输入，确定任务特征

        Args:
            user_input: 用户输入文本

        Returns:
            任务分析结果
        """
        # 1. 初步分类
        task_type = self._classify_task_type(user_input)

        # 2. 评估优先级
        priority = self._assess_priority(user_input, task_type)

        # 3. 评估复杂度
        complexity = self._assess_complexity(user_input, task_type)

        # 4. 生成推理过程
        reasoning = self._generate_reasoning(user_input, task_type, priority, complexity)

        # 5. 估算执行时间
        estimated_duration = self._estimate_duration(task_type, complexity)

        # 6. 确定所需能力
        required_capabilities = self._identify_capabilities(task_type, user_input)

        return TaskAnalysisResult(
            task_type=task_type,
            priority=priority,
            complexity=complexity,
            reasoning=reasoning,
            estimated_duration=estimated_duration,
            required_capabilities=required_capabilities
        )

    def _classify_task_type(self, user_input: str) -> TaskType:
        """
        基于关键词和语义分析确定任务类型

        Args:
            user_input: 用户输入

        Returns:
            任务类型
        """
        # 关键词匹配
        keyword_scores = {}
        input_lower = user_input.lower()

        for task_type, keywords in self.keyword_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in input_lower:
                    # 中文关键词权重更高
                    weight = 2 if any('\u4e00' <= c <= '\u9fff' for c in keyword) else 1
                    score += weight
            keyword_scores[task_type] = score

        # 如果有明确的关键词匹配，使用最高分类型
        max_score = max(keyword_scores.values())
        if max_score > 0:
            candidates = [t for t, s in keyword_scores.items() if s == max_score]
            return candidates[0]

        # 使用LLM进行语义分析
        return self._llm_classify_task_type(user_input)

    def _llm_classify_task_type(self, user_input: str) -> TaskType:
        """使用LLM进行任务类型分类"""
        prompt = f"""分析以下用户输入，并确定最合适的任务类型：

用户输入: {user_input}

任务类型选项:
- GENERAL_CONSULTATION: 一般咨询、解释、建议
- DATA_ANALYSIS: 数据分析、统计、报表
- PERFORMANCE_OPTIMIZATION: 性能优化、效率提升
- BUG_ANALYSIS: 错误分析、调试、修复
- CODE_REVIEW: 代码审查、质量评估
- DOCUMENTATION: 文档编写、说明撰写
- RESEARCH: 研究、调查、探索
- CREATIVE_WRITING: 创意写作、内容创作

请只返回任务类型名称，不要其他内容。"""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            response_text = response.content.strip().upper()

            # 尝试匹配任务类型
            for task_type in TaskType:
                if task_type.value.upper() in response_text:
                    return task_type

        except Exception as e:
            print(f"LLM分类失败: {e}")

        # 默认返回一般咨询
        return TaskType.GENERAL_CONSULTATION

    def _assess_priority(self, user_input: str, task_type: TaskType) -> TaskPriority:
        """
        评估任务优先级

        Args:
            user_input: 用户输入
            task_type: 任务类型

        Returns:
            任务优先级
        """
        input_lower = user_input.lower()

        # 高优先级关键词
        urgent_keywords = [
            '紧急', 'urgent', '立即', 'immediate', 'asap',
            '崩溃', 'crash', '宕机', 'down', '阻塞', 'blocking'
        ]

        # 中优先级关键词
        high_keywords = [
            '重要', 'important', '优先', 'priority',
            'bug', '错误', 'error', '修复', 'fix'
        ]

        # 检查紧急关键词
        for keyword in urgent_keywords:
            if keyword in input_lower:
                return TaskPriority.URGENT

        # 检查高优先级关键词
        for keyword in high_keywords:
            if keyword in input_lower:
                return TaskPriority.HIGH

        # 基于任务类型设置默认优先级
        type_priority_map = {
            TaskType.BUG_ANALYSIS: TaskPriority.HIGH,
            TaskType.PERFORMANCE_OPTIMIZATION: TaskPriority.HIGH,
            TaskType.DATA_ANALYSIS: TaskPriority.MEDIUM,
            TaskType.RESEARCH: TaskPriority.MEDIUM,
            TaskType.CODE_REVIEW: TaskPriority.MEDIUM,
            TaskType.CREATIVE_WRITING: TaskPriority.LOW,
            TaskType.DOCUMENTATION: TaskPriority.LOW,
            TaskType.GENERAL_CONSULTATION: TaskPriority.MEDIUM
        }

        return type_priority_map.get(task_type, TaskPriority.MEDIUM)

    def _assess_complexity(self, user_input: str, task_type: TaskType) -> str:
        """
        评估任务复杂度

        Args:
            user_input: 用户输入
            task_type: 任务类型

        Returns:
            复杂度级别: 'simple', 'medium', 'complex'
        """
        # 基础复杂度分数
        complexity_score = self.complexity_rules['task_type_complexity'].get(task_type, 1.0)

        # 基于输入长度调整
        if len(user_input) > self.complexity_rules['length_threshold']:
            complexity_score += 0.5

        # 基于关键词调整
        input_lower = user_input.lower()
        for keyword, weight in self.complexity_rules['keyword_complexity'].items():
            if keyword in input_lower:
                complexity_score += weight

        # 判断复杂度级别
        if complexity_score >= 2.5:
            return 'complex'
        elif complexity_score >= 1.5:
            return 'medium'
        else:
            return 'simple'

    def _generate_reasoning(self, user_input: str, task_type: TaskType,
                          priority: TaskPriority, complexity: str) -> str:
        """
        生成分析推理过程

        Args:
            user_input: 用户输入
            task_type: 任务类型
            priority: 优先级
            complexity: 复杂度

        Returns:
            推理过程说明
        """
        reasoning_parts = []

        # 任务类型推理
        type_reasons = {
            TaskType.DATA_ANALYSIS: "包含数据处理、统计分析相关的关键词",
            TaskType.PERFORMANCE_OPTIMIZATION: "涉及性能优化、效率提升的需求",
            TaskType.BUG_ANALYSIS: "包含错误调试、问题修复的描述",
            TaskType.CODE_REVIEW: "涉及代码质量评估和审查",
            TaskType.RESEARCH: "包含研究、调查、探索等活动",
            TaskType.CREATIVE_WRITING: "涉及内容创作和创意生成",
            TaskType.DOCUMENTATION: "涉及文档编写和说明撰写",
            TaskType.GENERAL_CONSULTATION: "一般的咨询和帮助请求"
        }

        reasoning_parts.append(f"任务类型: {type_reasons.get(task_type, '基于语义分析确定')}")

        # 优先级推理
        priority_reasons = {
            TaskPriority.URGENT: "包含紧急、崩溃、阻塞等关键词",
            TaskPriority.HIGH: "涉及重要问题或关键功能",
            TaskPriority.MEDIUM: "中等重要性和复杂度",
            TaskPriority.LOW: "较低优先级或简单任务"
        }

        reasoning_parts.append(f"优先级: {priority_reasons.get(priority, '基于任务类型和关键词确定')}")

        # 复杂度推理
        complexity_reasons = {
            'complex': "任务涉及多个步骤、需要深入分析或跨领域知识",
            'medium': "任务具有一定复杂度，需要专业知识",
            'simple': "任务相对简单，可以快速完成"
        }

        reasoning_parts.append(f"复杂度: {complexity_reasons.get(complexity, '基于输入长度和关键词分析')}")

        return "; ".join(reasoning_parts)

    def _estimate_duration(self, task_type: TaskType, complexity: str) -> int:
        """
        估算任务执行时间

        Args:
            task_type: 任务类型
            complexity: 复杂度

        Returns:
            预估时间（秒）
        """
        # 基础时间映射
        base_times = {
            TaskType.GENERAL_CONSULTATION: 60,
            TaskType.DATA_ANALYSIS: 180,
            TaskType.PERFORMANCE_OPTIMIZATION: 300,
            TaskType.BUG_ANALYSIS: 240,
            TaskType.CODE_REVIEW: 120,
            TaskType.DOCUMENTATION: 150,
            TaskType.RESEARCH: 200,
            TaskType.CREATIVE_WRITING: 120
        }

        base_time = base_times.get(task_type, 120)

        # 复杂度倍数
        complexity_multipliers = {
            'simple': 0.7,
            'medium': 1.0,
            'complex': 1.5
        }

        multiplier = complexity_multipliers.get(complexity, 1.0)

        return int(base_time * multiplier)

    def _identify_capabilities(self, task_type: TaskType, user_input: str) -> List[str]:
        """
        识别所需能力

        Args:
            task_type: 任务类型
            user_input: 用户输入

        Returns:
            所需能力列表
        """
        base_capabilities = {
            TaskType.DATA_ANALYSIS: ['data_processing', 'statistical_analysis'],
            TaskType.PERFORMANCE_OPTIMIZATION: ['performance_monitoring', 'optimization_techniques'],
            TaskType.BUG_ANALYSIS: ['debugging', 'error_analysis'],
            TaskType.CODE_REVIEW: ['code_quality_assessment', 'best_practices'],
            TaskType.RESEARCH: ['information_gathering', 'synthesis'],
            TaskType.CREATIVE_WRITING: ['content_creation', 'creativity'],
            TaskType.DOCUMENTATION: ['technical_writing', 'documentation'],
            TaskType.GENERAL_CONSULTATION: ['general_knowledge', 'explanation']
        }

        capabilities = base_capabilities.get(task_type, ['general_assistance'])

        # 基于输入内容添加额外能力
        input_lower = user_input.lower()

        if '代码' in input_lower or 'programming' in input_lower:
            capabilities.append('programming')
        if '数据' in input_lower or 'database' in input_lower:
            capabilities.append('database')
        if '网络' in input_lower or 'api' in input_lower:
            capabilities.append('networking')

        return list(set(capabilities))  # 去重

    async def assign_task(self, state: AgentState) -> AgentState:
        """
        执行任务分配

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        user_input = state.user_input

        # 分析任务
        analysis_result = await self.analyze_task(user_input)

        # 创建任务对象
        task = Task(
            id=f"task_{int(datetime.now().timestamp())}",
            user_input=user_input,
            task_type=analysis_result.task_type,
            task_description=user_input,
            priority=analysis_result.priority,
            complexity=analysis_result.complexity,
            requirements=[],  # 可以后续扩展
            expected_outcome="根据任务类型和复杂度提供相应的输出",
            max_retries=3,
            timeout_seconds=analysis_result.estimated_duration * 2,
            tags=[analysis_result.task_type.value, analysis_result.complexity],
            metadata={
                "analysis_result": analysis_result.model_dump(),
                "assigned_at": datetime.now().isoformat()
            }
        )

        # 更新状态
        updated_state = update_state_with_task_analysis(state, task)

        # 保存状态
        if self.state_manager:
            self.state_manager.update_state(
                updated_state.workflow_metadata.workflow_id,
                {"task": task.model_dump()}
            )

        return updated_state



