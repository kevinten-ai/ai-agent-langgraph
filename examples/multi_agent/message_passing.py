#!/usr/bin/env python3
"""
消息传递机制示例

这个示例展示了Agent间的消息传递和状态共享机制。
Agent之间通过结构化的消息进行通信和协作。
"""

import os
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 加载环境变量
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")


class AgentMessage(TypedDict):
    """Agent间传递的消息结构"""
    sender: str          # 发送者ID
    receiver: str        # 接收者ID
    message_type: str    # 消息类型: 'task', 'result', 'request', 'response', 'notification'
    content: str         # 消息内容
    timestamp: str       # 时间戳
    metadata: Dict[str, Any]  # 元数据


class MessagePassingState(TypedDict):
    """消息传递状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 对话历史
    agent_messages: Annotated[List[AgentMessage], operator.add]  # Agent间消息
    task: str                                            # 原始任务
    agent_states: Dict[str, Dict[str, Any]]            # 各Agent状态
    final_result: str                                   # 最终结果


class MessageBus:
    """消息总线 - 管理Agent间的消息传递"""

    def __init__(self):
        self.messages: List[AgentMessage] = []

    def send_message(self, sender: str, receiver: str, message_type: str,
                    content: str, metadata: Dict[str, Any] = None) -> None:
        """发送消息"""
        message: AgentMessage = {
            "sender": sender,
            "receiver": receiver,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        print(f"📨 {sender} -> {receiver}: {message_type} - {content[:50]}...")

    def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
        """获取发给特定Agent的消息"""
        return [msg for msg in self.messages if msg["receiver"] == agent_id]

    def get_unread_messages(self, agent_id: str, last_read_time: str = None) -> List[AgentMessage]:
        """获取未读消息"""
        messages = self.get_messages_for_agent(agent_id)
        if last_read_time:
            return [msg for msg in messages if msg["timestamp"] > last_read_time]
        return messages


class DataCollectorAgent:
    """数据收集Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def collect_data(self, task: str) -> str:
        """收集数据"""
        prompt = f"""你是一个数据收集专家。基于以下任务，收集相关信息：

任务: {task}

请提供：
1. 相关事实和数据
2. 关键指标和统计
3. 相关资源链接
4. 数据来源说明

保持客观和准确。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def process_message(self, message: AgentMessage) -> None:
        """处理接收到的消息"""
        if message["message_type"] == "task_assignment":
            print(f"🔍 {self.agent_id} 收到任务分配")
            task = message["content"]
            result = self.collect_data(task)

            # 发送结果给协调者
            self.message_bus.send_message(
                sender=self.agent_id,
                receiver="coordinator",
                message_type="task_result",
                content=result,
                metadata={"task_type": "data_collection", "original_task": task}
            )


class DataAnalyzerAgent:
    """数据分析Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    def analyze_data(self, data: str, task: str) -> str:
        """分析数据"""
        prompt = f"""你是一个数据分析专家。基于收集的数据，进行深入分析：

原始任务: {task}

收集的数据:
{data}

请提供：
1. 数据解读和洞察
2. 趋势识别和模式发现
3. 潜在影响评估
4. 建议和结论

用数据支持你的分析。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def process_message(self, message: AgentMessage) -> None:
        """处理接收到的消息"""
        if message["message_type"] == "data_ready":
            print(f"📊 {self.agent_id} 开始分析数据")
            data = message["content"]
            task = message["metadata"].get("original_task", "")
            result = self.analyze_data(data, task)

            # 发送分析结果给协调者
            self.message_bus.send_message(
                sender=self.agent_id,
                receiver="coordinator",
                message_type="analysis_result",
                content=result,
                metadata={"task_type": "data_analysis", "original_task": task}
            )


class ReportGeneratorAgent:
    """报告生成Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def generate_report(self, data: str, analysis: str, task: str) -> str:
        """生成报告"""
        prompt = f"""你是一个专业报告撰写者。基于数据收集和分析结果，生成完整报告：

原始任务: {task}

收集的数据:
{data}

分析结果:
{analysis}

生成一份结构化报告，包括：
1. 执行摘要
2. 方法说明
3. 主要发现
4. 分析洞察
5. 结论和建议

报告要专业、清晰、有说服力。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def process_message(self, message: AgentMessage) -> None:
        """处理接收到的消息"""
        if message["message_type"] == "analysis_complete":
            print(f"📝 {self.agent_id} 开始生成报告")
            data = message["metadata"].get("data", "")
            analysis = message["content"]
            task = message["metadata"].get("original_task", "")
            result = self.generate_report(data, analysis, task)

            # 发送最终报告给协调者
            self.message_bus.send_message(
                sender=self.agent_id,
                receiver="coordinator",
                message_type="final_report",
                content=result,
                metadata={"task_type": "report_generation", "original_task": task}
            )


def coordinator_node(state: MessagePassingState) -> MessagePassingState:
    """协调者节点 - 管理Agent间的协作"""
    message_bus = MessagePassingState.get("message_bus", MessageBus())
    task = state["task"]

    # 初始化Agent
    if "agents" not in state.get("agent_states", {}):
        collector = DataCollectorAgent("data_collector", message_bus)
        analyzer = DataAnalyzerAgent("data_analyzer", message_bus)
        reporter = ReportGeneratorAgent("report_generator", message_bus)

        agent_states = {
            "data_collector": {"agent": collector, "status": "idle"},
            "data_analyzer": {"agent": analyzer, "status": "idle"},
            "report_generator": {"agent": reporter, "status": "idle"}
        }
    else:
        agent_states = state["agent_states"]

    # 发送初始任务给数据收集Agent
    if not any(msg["message_type"] == "task_assignment" for msg in state.get("agent_messages", [])):
        message_bus.send_message(
            sender="coordinator",
            receiver="data_collector",
            message_type="task_assignment",
            content=task
        )

        # 模拟Agent处理消息
        collector = agent_states["data_collector"]["agent"]
        task_msg = message_bus.messages[-1]  # 获取刚发送的消息
        collector.process_message(task_msg)

    # 检查是否有新消息需要处理
    new_messages = []
    for agent_id, agent_info in agent_states.items():
        if agent_info["status"] != "completed":
            agent = agent_info["agent"]
            unread_messages = message_bus.get_messages_for_agent(agent_id)
            for msg in unread_messages:
                if msg not in [m for m in state.get("agent_messages", [])]:
                    agent.process_message(msg)
                    new_messages.append(msg)

                    # 根据消息类型更新Agent状态和触发后续操作
                    if msg["message_type"] == "task_result":
                        # 数据收集完成，触发分析
                        analyzer = agent_states["data_analyzer"]["agent"]
                        analysis_msg = message_bus.messages[-1]
                        analyzer.process_message({
                            **analysis_msg,
                            "receiver": "data_analyzer",
                            "message_type": "data_ready",
                            "metadata": {
                                **analysis_msg["metadata"],
                                "data": msg["content"]
                            }
                        })

                    elif msg["message_type"] == "analysis_result":
                        # 分析完成，触发报告生成
                        reporter = agent_states["report_generator"]["agent"]
                        report_msg = message_bus.messages[-1]
                        reporter.process_message({
                            **report_msg,
                            "receiver": "report_generator",
                            "message_type": "analysis_complete"
                        })

                    elif msg["message_type"] == "final_report":
                        # 报告生成完成
                        return {
                            "messages": [AIMessage(content="协作任务完成！")],
                            "agent_messages": new_messages,
                            "agent_states": agent_states,
                            "final_result": msg["content"]
                        }

    return {
        "messages": [AIMessage(content=f"协作进行中... 已处理 {len(new_messages)} 条消息")],
        "agent_messages": new_messages,
        "agent_states": agent_states,
        "final_result": state.get("final_result", "")
    }


def create_message_passing_graph():
    """创建消息传递协作图"""
    graph = StateGraph(MessagePassingState)

    # 添加节点
    graph.add_node("coordinator", coordinator_node)

    # 定义边
    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", END)  # 简化版，实际应该有条件判断

    return graph.compile()


def demonstrate_message_passing():
    """演示消息传递机制"""
    print("📨 Agent消息传递协作系统")
    print("系统包含：数据收集Agent、数据分析Agent、报告生成Agent")
    print("-" * 60)

    # 示例任务
    sample_tasks = [
        "分析当前人工智能在教育领域的应用情况",
        "评估电动汽车市场的发展趋势",
        "研究远程办公对企业文化的影响"
    ]

    print("可用的示例任务：")
    for i, task in enumerate(sample_tasks, 1):
        print(f"{i}. {task}")

    while True:
        user_input = input("\n请选择任务编号 (1-3) 或输入自定义任务: ").strip()

        if user_input in ['1', '2', '3']:
            task = sample_tasks[int(user_input) - 1]
        elif user_input.lower() in ['quit', 'exit', 'q']:
            print("📨 再见！")
            break
        elif user_input:
            task = user_input
        else:
            print("❌ 请输入有效任务")
            continue

        print(f"\n🚀 开始消息传递协作: {task}")
        print("=" * 60)

        try:
            # 创建协作系统
            collaboration_system = create_message_passing_graph()

            # 执行协作
            result = collaboration_system.invoke({
                "messages": [HumanMessage(content=f"任务：{task}")],
                "agent_messages": [],
                "task": task,
                "agent_states": {},
                "final_result": ""
            })

            # 显示结果
            print("\n📨 消息传递过程:")
            for msg in result.get("agent_messages", []):
                status = "📤" if msg["message_type"] in ["task_result", "analysis_result", "final_report"] else "📥"
                print(f"{status} {msg['sender']} -> {msg['receiver']}: {msg['message_type']}")

            if result.get("final_result"):
                print("\n📋 最终报告:")
                print("-" * 40)
                print(result["final_result"])

            print("\n✅ 协作完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        demonstrate_message_passing()
    except KeyboardInterrupt:
        print("\n📨 再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()



