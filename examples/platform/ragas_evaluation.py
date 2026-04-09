#!/usr/bin/env python3
"""
RAGAS 评估示例

构建一个最简单的 RAG pipeline（文档 + 向量检索 + 问答），
并使用 Ragas 对其进行评估。

运行前请确保已安装依赖：
    pip install ragas datasets langchain-openai langchain-community faiss-cpu

如果不用 OpenAI，也可以配置本地 LLM 或替换为其他 embedding。
"""

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
except ImportError as e:
    print("=" * 60)
    print(" ragas 未安装，请先执行以下命令安装依赖：")
    print("    pip install ragas datasets")
    print("=" * 60)
    raise SystemExit(0)

import os
from dotenv import load_dotenv

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# 1. 准备测试数据（少量样本，可直接运行）
# ============================================================
RAW_DOCUMENTS = [
    {
        "title": "LangGraph 介绍",
        "content": (
            "LangGraph 是由 LangChain 团队开发的一个用于构建\u591a Agent "
            "系统的框架。它基于图（Graph）结构，"
            "允许开发者将 LLM 应用构建为节点和边的组合，"
            "支持循环、条件分支和持久化状态。"
        ),
    },
    {
        "title": "LangGraph 状态管理",
        "content": (
            "LangGraph 使用 StateGraph 来定义工作流状态。"
            "每个节点都是一个函数，接收当前状态并返回更新后的状态。"
            "通过编译器，可以将这些节点组织成完整的应用。"
        ),
    },
    {
        "title": "RAGAS 评估",
        "content": (
            "RAGAS 是一套无需人工标注的 RAG 评估框架。"
            "它提供了 Faithfulness、Answer Relevancy 等指标，"
            "可以帮助开发者快速量化检索和生成质量。"
        ),
    },
]

# 两个问答对用于评估（答案由后面的 RAG pipeline 生成）
EVAL_QUESTIONS = [
    {
        "question": "LangGraph 是用来做什么的？",
        "ground_truth": "LangGraph 是用于构建多 Agent 系统的框架，基于图结构。",
    },
    {
        "question": "RAGAS 能评估哪些方面？",
        "ground_truth": "RAGAS 可以评估 Faithfulness 和 Answer Relevancy 等方面。",
    },
]


def build_rag_pipeline():
    """构建简易 RAG Pipeline：分割文档 → Embedding → FAISS 库 → 问答"""
    # 分割文档
    documents = [Document(page_content=d["content"], metadata={"title": d["title"]}) for d in RAW_DOCUMENTS]
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # 构建向量索引（使用小型 embedding 模型）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 创建 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def answer_question(question: str) -> dict:
        # 检索相关文档
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        contexts = retriever.invoke(question)
        context_text = "\n\n".join([c.page_content for c in contexts])

        # 构建 prompt 生成答案
        prompt = (
            "你是一个有帮助的助手。请严格根据以下上下文回答问题，"
            "如果上下文中没有相关信息，请回答'不知道'。\n\n"
            f"上下文：\n{context_text}\n\n"
            f"问题：{question}"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {
            "question": question,
            "answer": response.content,
            "contexts": [c.page_content for c in contexts],
        }

    return answer_question


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请在 .env 文件中设置 OPENAI_API_KEY")
        return

    print("🚀 正在构建 RAG Pipeline 并生成答案...")
    rag = build_rag_pipeline()

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for item in EVAL_QUESTIONS:
        result = rag(item["question"])
        questions.append(result["question"])
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        print(f"✅ 问题: {result['question']}")
        print(f"   答案: {result['answer']}")
        print(f"   上下文数: {len(result['contexts'])}")
        print()

    # 构建 Ragas 评估数据集
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("📊 开始运行 Ragas 评估...")
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_relevancy,
        ],
    )

    print("⭐ Ragas 评估结果:")
    for metric_name, score in result.items():
        print(f"   {metric_name}: {score:.4f}")


if __name__ == "__main__":
    main()
