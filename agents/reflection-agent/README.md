# Reflection Agent

A writer and a critic in a loop: one chain drafts a LinkedIn post, a second chain critiques it, and the critique is fed back as user feedback until the draft has been revised a few times.

**Skills:** `reflection` `self-critique` `iterative-refinement` `prompt-templates`
**Stack:** `LangGraph` `LangChain` `Ollama`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled.

```bash
cp .env.example .env   # LangSmith keys are optional, for tracing
uv sync && uv run main.py
```
