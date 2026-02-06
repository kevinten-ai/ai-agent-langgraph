# 配置说明

## 🔧 环境配置

### 1. 环境变量文件

创建 `.env` 文件并配置以下变量：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# LangChain 配置（可选）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key

# 项目配置
LOG_LEVEL=INFO
MAX_TOKENS=2000
TEMPERATURE=0.7
```

### 2. API密钥获取

1. 访问 [OpenAI平台](https://platform.openai.com/)
2. 创建API密钥
3. 将密钥添加到 `.env` 文件中

### 3. 权限设置

确保项目目录有适当的读写权限：

```bash
# Linux/Mac
chmod 644 config/.env

# Windows (PowerShell)
icacls config\.env /grant:r "$env:USERNAME:R"
```

## 📦 依赖安装

### 基础依赖

```bash
pip install -r requirements.txt
```

### 可选依赖

```bash
# 包含工具支持
pip install crewai[tools]

# 包含所有可选功能
pip install langchain[all] pandas scikit-learn
```

## 🚀 运行示例

### 1. 基础Agent

```bash
cd examples/basic_agent
python simple_chatbot.py
```

### 2. 多Agent协作

```bash
cd examples/multi_agent
python role_based_agents.py
```

### 3. 复杂工作流

```bash
cd examples/complex_workflow
python conditional_flows.py
```

## 🧪 测试配置

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_basic_agent.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试配置

在 `pytest.ini` 中配置测试选项：

```ini
[tool:pytest.ini_options]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

## 📊 监控配置

### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent.log'),
        logging.StreamHandler()
    ]
)
```

### 性能监控

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 🔒 安全配置

### API密钥安全

1. 永远不要将 `.env` 文件提交到版本控制
2. 使用环境变量而不是硬编码密钥
3. 定期轮换API密钥
4. 限制API密钥权限

### 数据安全

1. 不要在日志中记录敏感信息
2. 使用加密存储敏感数据
3. 实施适当的访问控制
4. 定期清理临时文件

## 🌐 部署配置

### 本地部署

```bash
# 开发模式
python -m uvicorn app:app --reload

# 生产模式
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 云部署

- **Vercel**: 无服务器部署
- **Railway**: 简单云部署
- **AWS Lambda**: 事件驱动部署
- **Google Cloud Run**: 容器化部署

## 🔧 故障排除

### 常见问题

1. **API密钥错误**
   - 检查 `.env` 文件是否存在
   - 验证API密钥格式
   - 确认账户余额充足

2. **依赖安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用虚拟环境
   - 检查Python版本兼容性

3. **网络连接问题**
   - 检查防火墙设置
   - 验证代理配置
   - 尝试更换网络环境

### 调试技巧

1. 启用详细日志
2. 使用断点调试
3. 检查网络请求
4. 验证数据格式

## 📞 支持

如果遇到配置问题，请：

1. 检查本文档
2. 查看GitHub Issues
3. 提交问题报告
4. 联系技术支持



