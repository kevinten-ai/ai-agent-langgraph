# AI Agent LangGraph Maintenance Guide

## Project State

- Repository: `kevinten-ai/ai-agent-langgraph`
- Category: LangChain/LangGraph learning and production-pattern reference
- Runtime: Python examples, LangGraph server example, Docker Compose deployment path
- Main docs: `docs/00-learning-roadmap.md` through `docs/09-production-deployment.md`

## Before Editing

- Keep examples runnable independently where possible.
- Do not spend real LLM or tracing quota in default tests.
- Keep API keys in local `.env` files or platform secrets; do not commit filled credentials.

## Useful Commands

- Create environment: `python -m venv .venv`
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest`
- Run demo: `python demo.py`
- LangGraph server example: see `examples/platform/langgraph_server/`
- Docker compose smoke check: `docker compose config`

## Environment

- Root `.env.example` mirrors the key variables needed by the examples.
- `config/.env.example` remains the server-specific reference used by existing docs.

## Deployment Notes

- See `DEPLOYMENT.md` for LangGraph server and Docker Compose checks.
- Keep the package-style `langgraph.json` module path in sync when workflow modules move.
