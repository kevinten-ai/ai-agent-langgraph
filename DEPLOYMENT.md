# AI Agent LangGraph Deployment Notes

## Current Status

- Repository: `kevinten-ai/ai-agent-langgraph`
- Deployment examples: `Dockerfile`, `docker-compose.yml`, `langgraph.json`
- Primary purpose: learning/reference plus production-pattern examples.

## Local Validation

```bash
python -m venv .venv
pip install -r requirements.txt
python -m compileall -q demo.py src examples
docker compose config
```

## LangGraph Server Path

- Keep the package-style `langgraph.json` module path aligned with the workflow module;
  a direct file path breaks the workflow's relative imports.
- Use local or disposable API keys for smoke tests.
- Disable tracing by default unless validating observability examples.

## Deployment Checklist

- Provide `OPENAI_API_KEY` or an OpenAI-compatible provider key through platform secrets.
- Confirm token and request-cost behavior before running long multi-agent examples.
- Verify health and graph invocation endpoints in the target LangGraph server environment.
- Keep example datasets and generated traces out of Git.
