# langgraph_server 部署示例

本目录汇集了 LangGraph 学习项目的核心平台化部署配置，用于演示如何将本地开发的图（Graph）通过 Docker 运行起来，并接入 Redis + PostgreSQL 的完整平台栈。

## 文件说明

| 文件 | 用途 |
|------|------|
| `../../Dockerfile` | 构建 LangGraph 运行镜像（多阶段构建） |
| `../../docker-compose.yml` | 编排 Redis、Postgres、LangGraph API 三个服务 |
| `../../langgraph.json` | LangGraph Server 入口配置，声明图的加载路径 |
| `../../.dockerignore` | 控制哪些文件不进入 Docker 构建上下文 |

## 快速启动

在项目根目录执行：

```bash
docker-compose up --build
```

首次构建会安装依赖并启动三个容器；后续若未修改 `requirements.txt` 或 `Dockerfile`，可直接用：

```bash
docker-compose up
```

## 访问地址

- LangGraph API: http://localhost:8123
- 健康检查: http://localhost:8123/ok

## 环境变量说明

环境变量分三层管理：

### 1. 本地敏感配置（.env 文件）
仓库根目录下的 `config/.env` 存放 API Key 等敏感信息，**已加入 .gitignore，请勿提交**。最小必填项：

```bash
OPENAI_API_KEY=sk-xxx
```

### 2. docker-compose 注入的公共变量
在 `docker-compose.yml` 的 `langgraph-api` 服务中已预置：

| 变量名 | 说明 |
|--------|------|
| `LANGGRAPH_API_URL` | 服务自身暴露的地址 |
| `REDIS_URI` | Redis 连接串（平台缓存） |
| `POSTGRES_URI` | PostgreSQL 连接串（Checkpointer 持久化） |
| `LANGCHAIN_TRACING_V2` | 是否开启 LangSmith 跟踪（本示例默认 `false`） |

### 3. 容器编排相关

- **Redis**: 暴露在 `6379` 端口
- **Postgres**: 暴露在 `5432` 端口
  - 数据库名: `langgraph`
  - 用户名: `langgraph`
  - 密码: `langgraph`
- **LangGraph API**: 暴露在 `8123` 端口

## 停止服务

```bash
# 正常停止（保留数据）
docker-compose down

# 完全清空持久化数据（慎用）
docker-compose down -v
```

## 学习提示

- 这是教学项目的**本地自托管**演示配置，目的是理解平台部署的全貌。
- 若需要正式上云，可考虑 [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/)。
