"""
状态管理器

提供状态的创建、更新、验证、序列化和持久化功能。
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from ..models import AgentState, WorkflowMetadata, get_state_summary


class StateManager:
    """状态管理器"""

    def __init__(self, max_workers: int = 4):
        self._states: Dict[str, AgentState] = {}
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._state_listeners: List[callable] = []

    def create_state(self, user_input: str, workflow_id: Optional[str] = None) -> AgentState:
        """创建新状态"""
        from ..models import create_initial_state

        if not workflow_id:
            workflow_id = f"wf_{int(datetime.now().timestamp())}"

        state = create_initial_state(user_input)
        state.workflow_metadata.workflow_id = workflow_id

        with self._lock:
            self._states[workflow_id] = state
            self._notify_listeners("created", state)

        return state

    def get_state(self, workflow_id: str) -> Optional[AgentState]:
        """获取状态"""
        with self._lock:
            return self._states.get(workflow_id)

    def update_state(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[AgentState]:
        """更新状态"""
        with self._lock:
            state = self._states.get(workflow_id)
            if not state:
                return None

            # 更新状态字段
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            # 更新时间戳
            if hasattr(state, 'workflow_metadata'):
                state.workflow_metadata.updated_at = datetime.now()

            # 通知监听器
            self._notify_listeners("updated", state)

            return state

    def delete_state(self, workflow_id: str) -> bool:
        """删除状态"""
        with self._lock:
            if workflow_id in self._states:
                state = self._states[workflow_id]
                del self._states[workflow_id]
                self._notify_listeners("deleted", state)
                return True
        return False

    def list_states(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有状态摘要"""
        with self._lock:
            states = []
            for state in self._states.values():
                summary = get_state_summary(state)
                if not status_filter or summary["status"] == status_filter:
                    states.append(summary)
            return states

    def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """清理旧状态"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0

        with self._lock:
            to_remove = []
            for workflow_id, state in self._states.items():
                if state.workflow_metadata.start_time.timestamp() < cutoff_time:
                    to_remove.append(workflow_id)

            for workflow_id in to_remove:
                del self._states[workflow_id]
                cleaned_count += 1

        return cleaned_count

    def add_listener(self, listener: callable):
        """添加状态监听器"""
        self._state_listeners.append(listener)

    def remove_listener(self, listener: callable):
        """移除状态监听器"""
        if listener in self._state_listeners:
            self._state_listeners.remove(listener)

    def _notify_listeners(self, event: str, state: AgentState):
        """通知监听器"""
        for listener in self._state_listeners:
            try:
                listener(event, state)
            except Exception as e:
                # 监听器错误不应该影响主流程
                print(f"状态监听器错误: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取状态统计信息"""
        with self._lock:
            total_states = len(self._states)
            status_counts = {}
            agent_counts = {}

            for state in self._states.values():
                status = state.workflow_status
                agent = state.current_agent

                status_counts[status] = status_counts.get(status, 0) + 1
                if agent:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1

            return {
                "total_states": total_states,
                "status_distribution": status_counts,
                "agent_distribution": agent_counts,
                "memory_usage": self._estimate_memory_usage()
            }

    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        # 简单估算：每个状态大约1KB
        return len(self._states) * 0.001


class StateValidator:
    """状态验证器"""

    @staticmethod
    def validate_state(state: AgentState) -> Dict[str, Any]:
        """验证状态完整性"""
        required_fields = ["user_input", "workflow_status"]
        missing_fields = []
        validation_errors = []

        # 检查必需字段
        for field in required_fields:
            if not getattr(state, field, None):
                missing_fields.append(field)

        # 检查状态一致性
        if state.workflow_status == "completed" and not state.execution_result:
            validation_errors.append("已完成的工作流必须有执行结果")

        if state.needs_retry and state.review_score and state.review_score > 7.0:
            validation_errors.append("高分审核结果不应需要重试")

        # 检查数据类型
        if state.review_score is not None and not (0.0 <= state.review_score <= 10.0):
            validation_errors.append("审核分数必须在0-10之间")

        return {
            "is_valid": len(missing_fields) == 0 and len(validation_errors) == 0,
            "missing_fields": missing_fields,
            "validation_errors": validation_errors,
            "completeness_score": StateValidator._calculate_completeness(state)
        }

    @staticmethod
    def validate_transition(from_state: AgentState, to_state: AgentState,
                          transition: str) -> bool:
        """验证状态转换"""
        transition_rules = {
            "task_assigner->executor": lambda f, t: (
                t.task_type is not None and
                t.task_description is not None
            ),
            "executor->reviewer": lambda f, t: (
                t.execution_result is not None
            ),
            "reviewer->end": lambda f, t: (
                t.review_score is not None
            )
        }

        validator = transition_rules.get(transition)
        if validator:
            return validator(from_state, to_state)

        return True

    @staticmethod
    def _calculate_completeness(state: AgentState) -> float:
        """计算状态完整性分数"""
        fields = [
            "user_input", "task", "execution_result",
            "review_score", "workflow_metadata"
        ]

        completed_fields = sum(1 for field in fields if getattr(state, field, None) is not None)
        return completed_fields / len(fields)


class StateSerializer:
    """状态序列化器"""

    @staticmethod
    def serialize(state: AgentState) -> str:
        """序列化状态"""
        # 转换为字典
        state_dict = state.model_dump()

        # 处理特殊对象
        StateSerializer._preprocess_for_json(state_dict)

        # JSON序列化
        return json.dumps(state_dict, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def deserialize(json_str: str) -> AgentState:
        """反序列化状态"""
        state_dict = json.loads(json_str)

        # 处理特殊对象
        StateSerializer._postprocess_from_json(state_dict)

        # 创建状态对象
        return AgentState(**state_dict)

    @staticmethod
    def save_to_file(state: AgentState, file_path: str) -> bool:
        """保存状态到文件"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json_str = StateSerializer.serialize(state)
                f.write(json_str)

            return True
        except Exception as e:
            print(f"保存状态失败: {e}")
            return False

    @staticmethod
    def load_from_file(file_path: str) -> Optional[AgentState]:
        """从文件加载状态"""
        try:
            if not Path(file_path).exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                json_str = f.read()

            return StateSerializer.deserialize(json_str)
        except Exception as e:
            print(f"加载状态失败: {e}")
            return None

    @staticmethod
    def _preprocess_for_json(data: Dict[str, Any]):
        """JSON序列化预处理"""
        # 处理枚举类型
        for key, value in data.items():
            if hasattr(value, 'value'):  # 枚举
                data[key] = value.value
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, dict):
                StateSerializer._preprocess_for_json(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        StateSerializer._preprocess_for_json(item)

    @staticmethod
    def _postprocess_from_json(data: Dict[str, Any]):
        """JSON反序列化后处理"""
        # 这里可以添加反序列化后的转换逻辑
        # 例如转换字符串到枚举等
        pass


class StatePersistence:
    """状态持久化管理器"""

    def __init__(self, storage_dir: str = "./data/states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: AgentState) -> bool:
        """保存状态"""
        workflow_id = state.workflow_metadata.workflow_id
        file_path = self.storage_dir / f"{workflow_id}.json"
        return StateSerializer.save_to_file(state, str(file_path))

    def load_state(self, workflow_id: str) -> Optional[AgentState]:
        """加载状态"""
        file_path = self.storage_dir / f"{workflow_id}.json"
        return StateSerializer.load_from_file(str(file_path))

    def delete_state(self, workflow_id: str) -> bool:
        """删除状态"""
        file_path = self.storage_dir / f"{workflow_id}.json"
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception as e:
            print(f"删除状态文件失败: {e}")
        return False

    def list_saved_states(self) -> List[str]:
        """列出保存的状态"""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """清理旧文件"""
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        for file_path in self.storage_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                cleaned_count += 1

        return cleaned_count



