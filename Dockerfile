# ================================================
# LangGraph Standalone Server - 学习项目用 Dockerfile
# 目标: 构建可直接 `docker run` 的轻量教学镜像
# ================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 安装 langgraph-cli[inmem]（包含 CLI 与本地内存服务器依赖）
RUN pip install --no-cache-dir --user "langgraph-cli[inmem]"

# ================================================
# 运行阶段（多阶段构建，减少镜像体积）
# ================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制项目文件
COPY . .

# 暴露 LangGraph Server 默认端口
EXPOSE 8123

# 默认启动开发服务器（教学演示用）
# 生产环境建议替换为: CMD ["langgraph", "up", "--host", "0.0.0.0", "--port", "8123"]
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--no-browser"]
