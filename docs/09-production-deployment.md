# 09 生产部署指南

> 目标：把 LangGraph 学习项目从本地代码运行平滑迁移到可自托管、可上云的部署形态。

---

## 1. 部署方式对比

| 维度 | `langgraph dev` | `docker-compose up` | LangGraph Cloud |
|------|-----------------|---------------------|-----------------|
| **适用场景** | 本地开发、调试图逻辑 | 本地/服务器自托管、演示、轻量生产 | 企业级生产、自动扩缩容 |
| **部署复杂度** | 极低（一条命令） | 低（需安装 Docker） | 中（需配置 Cloud 项目） |
| **持久化能力** | 默认内存（可接 SQLite） | Redis + Postgres（本指南已集成） | 托管 Redis/Postgres |
| **CI/CD** | 手动执行 | 可结合 GitHub Actions 自部署 | 原生集成 Git 自动部署 |
| **成本** | 免费 | 服务器费用自理 | 按调用量/实例计费 |
| **学习建议** | 先用它写图 | 再用它理解“完整平台运行态” | 最后了解企业级选项 |

**推荐学习路径**：
1. 先用 `langgraph dev` 把图画对。
2. 再用 `docker-compose up` 体验“平台化运行”（包含健康检查、持久化、环境变量管理）。
3. 若后续需要流量弹性、自动版本回滚，再评估 LangGraph Cloud。

---

## 2. `langgraph.json` 配置项详解

`langgraph.json` 是 LangGraph Server 的**入口清单**，告诉 CLI/Cloud 如何加载你的图。

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/workflow/orchestrator.py:get_graph"
  },
  "env": "./config/.env",
  "python_version": "3.11",
  "pip_config": {
    "extra_index_urls": []
  }
}
```

| 字段 | 含义 | 学习提示 |
|------|------|----------|
| `dependencies` | 项目包路径列表。`"."` 表示把当前目录当成一个 Python 包安装。 | 确保仓库根目录有可被 `pip install -e .` 识别的结构。 |
| `graphs` | 定义图的名称到“模块路径:工厂函数”的映射。 | `agent` 是图 ID，可在 API/SDK 中引用它。 |
| `env` | 环境变量文件路径。 | 不要把 `.env` 提交到 Git，这在 Dockerfile 中也会被拷贝。 |
| `python_version` | 指定运行时的 Python 主版本。 | 要与 Dockerfile 基础镜像保持一致。 |
| `pip_config` | 额外的 pip 源配置。 | 若使用私有 PyPI 或企业镜像，在这里配置。 |

---

## 3. Dockerfile 多阶段构建最佳实践

本项目的 Dockerfile 采用**多阶段构建**（Multi-stage Build），目的有两个：
- **减少最终镜像体积**：编译依赖放在 `builder` 阶段，运行阶段只保留安装好的包。
- **提升构建速度**：`requirements.txt` 不变时，Docker 会复用缓存层。

```dockerfile
# 阶段 1：builder（安装依赖）
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user "langgraph-cli[inmem]"

# 阶段 2：runtime（仅运行）
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .
EXPOSE 8123
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--no-browser"]
```

### 学习要点
- `--user` 安装到 `/root/.local`，避免污染系统 Python。
- `langgraph-cli[inmem]` 同时安装 CLI 与**内存模式**的服务器依赖，适合教学 demo。
- 若进入真实生产，建议把 `CMD` 改为 `langgraph up` 或自定义 ASGI 入口。

---

## 4. 环境变量与密钥管理

### 4.1 环境变量分层

| 层级 | 文件/位置 | 用途 |
|------|-----------|------|
| 代码默认值 | Python 代码 | 非敏感、可在仓库中保留 |
| 本地开发 | `config/.env` | OPENAI_API_KEY、调试开关 |
| Docker Compose | `docker-compose.yml` | 服务间连接字符串（如 Redis/Postgres URI） |
| 生产服务器 | 宿主环境变量 / Secret Manager | 所有密钥和连接信息 |

### 4.2 密钥管理原则

1. **绝不硬编码**：API Key、数据库密码必须走环境变量。
2. **不进 Git**：`.env`、`config/.env` 已在 `.gitignore` 中排除。
3. **最小权限**：给 LangGraph 服务的数据库账号只开必要权限（读/写 checkpoint 表即可）。
4. **生产旋转**：定期轮换 API Key 和数据库密码。

### 4.3 常用环境变量示例

```bash
# LLM
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1

# LangSmith（可选）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-xxx
LANGCHAIN_PROJECT=ai-agent-langgraph

# 平台连接（docker-compose 已自动注入）
REDIS_URI=redis://redis:6379/0
POSTGRES_URI=postgresql://langgraph:langgraph@postgres:5432/langgraph?sslmode=disable
```

---

## 5. 使用 docker-compose 的一步启动教程

### 5.1 前置条件

- 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop) 或 Docker Engine + Compose。
- 在项目根目录准备好 `config/.env`，至少写入 `OPENAI_API_KEY`。

### 5.2 启动命令

```bash
# 进入项目根目录
cd ai-agent-langgraph

# 一键构建并启动所有服务
docker-compose up --build
```

首次构建会稍慢（安装 Python 依赖），后续若 `requirements.txt` 未变会复用缓存。

### 5.3 验证服务

```bash
# 检查健康状态
curl http://localhost:8123/ok

# 查看运行中的容器
docker-compose ps
```

### 5.4 停止命令

```bash
# 前台停止：Ctrl + C
# 后台停止：
docker-compose down

# 如需清空持久化数据（谨慎）：
docker-compose down -v
```

---

## 6. Health Check / Graceful Shutdown 概念

### 6.1 Health Check（健康检查）

`docker-compose.yml` 中为三个服务都配置了 `healthcheck`：

- **Redis**：`redis-cli ping` —— 确认缓存可连接。
- **Postgres**：`pg_isready` —— 确认数据库就绪。
- **langgraph-api**：`curl http://localhost:8123/ok` —— 确认 LangGraph Server 已启动并响应请求。

`depends_on` 配合 `condition: service_healthy` 保证：
**LangGraph 应用会在 Redis 和 Postgres 都健康后才启动**，避免“数据库还没好，应用就崩溃”的启动时序问题。

### 6.2 Graceful Shutdown（优雅关闭）

优雅关闭指容器收到 `SIGTERM`（停止信号）时，应用不是直接被杀掉，而是：
1. 停止接收新请求。
2. 完成正在处理的图执行。
3. 关闭数据库连接。
4. 再退出进程。

**学习提示**：LangGraph CLI 内置了基本的优雅关闭支持。若你未来编写自定义 ASGI 服务器，建议显式监听 `SIGTERM` 信号，给进行中的 `ainvoke` / `astream` 留出完成时间。

---

## 7. 本节总结

- `langgraph.json` 是图的“身份证”，声明了入口工厂函数和运行时信息。
- Dockerfile 采用多阶段构建，兼顾教学简洁性与镜像体积控制。
- `docker-compose.yml` 把 Redis、Postgres、LangGraph API 编排在一起，是一次“平台化”运行的最小可运行单元。
- 环境变量+`.gitignore` 构成本地到生产的安全基线。
- Health Check 和 Graceful Shutdown 是云原生部署的基本素养。

---

**相关文件**：

- `/Users/kevinten/projects/langchain/ai-agent-langgraph/langgraph.json`
- `/Users/kevinten/projects/langchain/ai-agent-langgraph/Dockerfile`
- `/Users/kevinten/projects/langchain/ai-agent-langgraph/docker-compose.yml`
- `/Users/kevinten/projects/langchain/ai-agent-langgraph/.dockerignore`
- `/Users/kevinten/projects/langchain/ai-agent-langgraph/examples/platform/langgraph_server/README.md`
