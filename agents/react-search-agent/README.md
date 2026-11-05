# ReAct Search Agent

Web research agent that searches the internet and returns a typed answer with its source URLs.

**Skills:** `ReAct-agents` `tool-calling` `structured-output`
**Stack:** `LangChain` `Ollama` `Tavily` `Pydantic`

## Run

Requires [Ollama](https://ollama.com) with `qwen3.8` pulled.

```bash
cp .env.example .env
uv sync && uv run main.py
```
