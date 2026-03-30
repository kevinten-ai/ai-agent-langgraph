# LangChain 核心概念

> LangChain 的设计哲学：**一切皆 Runnable**。模型、提示词、解析器、检索器、工具——都实现同一个接口，通过管道符自由组合。

## 一、分层架构

![LangChain 分层架构](./images/01-langchain-architecture.png)

```
┌─────────────────────────────────────────────────────┐
│               你的应用 (Application)                  │
├─────────────────────────────────────────────────────┤
│  langchain              │  langgraph               │
│  (chains, agents,       │  (StateGraph,            │
│   middleware)            │   multi-agent)           │
├─────────────────────┴───────────────────────────────┤
│                  langchain-core                      │
│   Runnable │ LCEL │ BaseChatModel │ BaseTool │      │
│   BaseRetriever │ Messages │ OutputParser           │
├─────────────────────────────────────────────────────┤
│  langchain-openai  │ langchain-anthropic  │ ...     │
│              (Provider 集成实现层)                     │
└─────────────────────────────────────────────────────┘
```

### 包结构

| 包名 | 角色 | 版本 |
|------|------|------|
| **`langchain-core`** | 基础抽象层 (Runnable, LCEL, 基类) | 1.2.x |
| **`langchain`** | 高级 API (chains, agents, middleware) | 1.0.x |
| **`langchain-community`** | 第三方社区集成 | 0.3.x |
| **Provider 包** | 一等集成 (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`) | 各自独立 |
| **`langgraph`** | 图编排框架 | 1.0.x |
| **`langgraph-checkpoint`** | 持久化后端 (MemorySaver, SqliteSaver, PostgresSaver) | 独立发布 |

**核心原则：`langchain-core` 定义接口，Provider 包实现接口，`langchain` 和 `langgraph` 编排接口。**

---

## 二、核心抽象

### 1. Runnable — 万物基石

`Runnable[Input, Output]` 是整个框架的基础协议。所有组件都实现它：

```python
# 统一接口 — 无论是模型、提示词还是解析器
result = component.invoke(input)           # 单次调用
results = component.batch([in1, in2])      # 批量调用
for chunk in component.stream(input):      # 流式输出
    print(chunk)

# 异步版本
result = await component.ainvoke(input)
async for chunk in component.astream(input):
    print(chunk)

# 高级：流式事件（用于复杂链的细粒度监控）
async for event in component.astream_events(input):
    print(event)
```

**声明式修饰器：**

```python
# 绑定参数
model_with_temp = model.bind(temperature=0.5)

# 重试策略
resilient = model.with_retry(max_retries=3)

# 降级方案
safe = model.with_fallbacks([backup_model])

# 运行时可配置
configurable = model.configurable_fields(temperature=ConfigurableField(id="temp"))
```

**Schema 内省：**

```python
chain.input_schema    # 输入的 JSON Schema
chain.output_schema   # 输出的 JSON Schema
chain.get_graph()     # 获取执行图（可视化）
```

> **关键理解：** Runnable 协议意味着你可以对任何组件做 invoke/stream/batch，不需要关心它内部是什么。这是 LangChain 的"一切皆 Runnable"哲学。

---

### 2. ChatModel — 模型调用

Provider 无关的 LLM 接口。无论底层是 GPT-4o、Claude、Gemini 还是 Llama，调用方式完全一致：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 创建模型实例
model = ChatOpenAI(model="gpt-4o")
# model = ChatAnthropic(model="claude-sonnet-4-20250514")  # 切换 Provider 只需换一行

# 输入: 消息列表 → 输出: AIMessage
response = model.invoke([
    SystemMessage(content="你是一个Python专家"),
    HumanMessage(content="什么是装饰器？")
])

print(response.content)       # 文本内容
print(response.tool_calls)    # 工具调用（如果有）
print(response.usage_metadata) # Token 使用统计
```

**消息类型：**

| 类型 | 说明 |
|------|------|
| `SystemMessage` | 系统提示，设定角色和规则 |
| `HumanMessage` | 用户消息 |
| `AIMessage` | 模型回复 |
| `ToolMessage` | 工具执行结果（回传给模型） |

**模型绑定工具：**

```python
# 让模型知道可以调用哪些工具
model_with_tools = model.bind_tools([search_tool, calculator_tool])
```

---

### 3. PromptTemplate — 提示词模板

将 prompt engineering 变成参数化、版本可控的组件：

```python
from langchain_core.prompts import ChatPromptTemplate

# 基础模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}专家，用{language}回答"),
    ("human", "{question}")
])

# 调用（实现了 Runnable 接口）
messages = prompt.invoke({
    "role": "Python",
    "language": "中文",
    "question": "什么是装饰器？"
})
```

**Few-shot 模板：**

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3*5", "output": "15"},
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    examples=examples,
)
```

---

### 4. OutputParser — 结构化输出

将 LLM 的自由文本输出解析为结构化 Python 对象：

```python
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# 字符串解析（最简单）
parser = StrOutputParser()

# JSON 解析
class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分 1-10")
    summary: str = Field(description="一句话总结")

json_parser = JsonOutputParser(pydantic_object=MovieReview)

# 更推荐：模型原生结构化输出
structured_model = model.with_structured_output(MovieReview)
result = structured_model.invoke("评价一下《盗梦空间》")
# result 是 MovieReview 实例
```

---

### 5. LCEL — 管道组合

![LCEL 管道概念](./images/02-lcel-pipeline.png)

LangChain Expression Language 用 `|` 管道符组合 Runnable：

```python
# 最核心的范式：prompt → model → parser
chain = prompt | model | parser
result = chain.invoke({"role": "Python", "question": "什么是装饰器"})
```

**LCEL 不是语法糖，而是创建了 `RunnableSequence`。** 整条链自动获得 `stream()`、`batch()`、`ainvoke()` 能力。

**组合原语：**

```python
from langchain_core.runnables import (
    RunnableSequence,      # | 操作符创建，从左到右执行
    RunnableParallel,      # 字典字面量创建，并行执行
    RunnablePassthrough,   # 原样传递输入
    RunnableLambda,        # 包装任意 Python 函数
)

# 并行分支
chain = RunnableParallel({
    "summary": prompt_summary | model | parser,
    "keywords": prompt_keywords | model | parser,
})
# 输入同时流入两条分支，结果合并为 dict

# 保留原始输入 + 添加处理结果
chain = RunnableParallel({
    "original": RunnablePassthrough(),
    "processed": some_chain,
})

# 包装自定义函数
def format_output(text: str) -> str:
    return text.upper()

chain = prompt | model | StrOutputParser() | RunnableLambda(format_output)
```

**流式传输：**

```python
# 链在 stream() 时自动逐 token 流式传输
for chunk in chain.stream({"question": "解释量子计算"}):
    print(chunk, end="", flush=True)
```

---

## 三、RAG 检索增强生成

![RAG Pipeline](./images/06-rag-pipeline.png)

RAG (Retrieval-Augmented Generation) 是 LangChain 最重要的应用模式之一：

```
用户提问 → 检索相关文档 → 将文档作为上下文注入 Prompt → LLM 生成回答
```

### 核心组件

```python
# 1. 文档加载
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()  # → List[Document]

# 2. 文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. 向量嵌入
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 4. 向量存储
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. 检索器（实现 Runnable 接口）
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 6. RAG 链
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题：\n\n{context}"),
    ("human", "{question}")
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("这篇文档讲了什么？")
```

### Retriever 接口

```python
from langchain_core.retrievers import BaseRetriever

class MyRetriever(BaseRetriever):
    """自定义检索器 — 实现 Runnable 接口"""

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # 你的检索逻辑
        return [Document(page_content="...")]
```

---

## 四、Agent 智能体

Agent = LLM + Tools + 决策循环。LLM 决定调用哪个工具、传什么参数、何时结束。

### 工具定义

```python
from langchain_core.tools import tool, BaseTool

# 最简方式：@tool 装饰器
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""  # docstring 会作为工具描述给 LLM
    return f"搜索结果: {query}"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，返回计算结果"""
    # 生产环境应使用安全的数学表达式解析库如 numexpr 或 sympy
    import numexpr
    return str(numexpr.evaluate(expression))
```

### 创建 Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
tools = [search_web, calculator]

# create_agent 内部使用 LangGraph 构建了一个 ReAct 循环
agent = create_agent(model, tools)

# 标准 Runnable 接口
result = agent.invoke({"messages": [("human", "北京今天天气怎么样？")]})
```

### ReAct 循环（Agent 核心模式）

```
用户提问
   │
   ▼
┌─────────────┐
│  LLM 思考    │ ← "我需要搜索天气信息"
└──────┬──────┘
       │ tool_call: search_web("北京天气")
       ▼
┌─────────────┐
│  执行工具    │ ← 实际调用搜索
└──────┬──────┘
       │ ToolMessage: "北京今天晴，25°C"
       ▼
┌─────────────┐
│  LLM 思考    │ ← "我已经有了足够信息"
└──────┬──────┘
       │ 直接回复（不再调用工具）
       ▼
   最终回答
```

### 结构化输出

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

# 模型直接输出 Pydantic 对象
structured_model = model.with_structured_output(AnalysisResult)
result = structured_model.invoke("分析这段文本的情感...")
# result.sentiment, result.confidence, result.keywords
```

---

## 五、回调与可观测性

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 开始调用...")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM 完成，Token: {response.llm_output}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"工具调用: {input_str}")

# 使用回调
result = chain.invoke(input, config={"callbacks": [MyCallback()]})
```

**LangSmith 集成：**

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls_..."
# 自动追踪所有 chain/agent 调用，无需改代码
```
