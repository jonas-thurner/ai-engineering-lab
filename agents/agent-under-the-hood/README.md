# Agent Under the Hood

A tool-calling agent loop written by hand, without an agent framework, to show what an agent actually does on every iteration.

**Skills:** `tool-calling` `agent-loop` `tracing`
**Stack:** `LangChain` `Ollama` `LangSmith`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled.

```bash
cp .env.example .env   # LangSmith keys are optional, for tracing
uv sync && uv run main.py
```
