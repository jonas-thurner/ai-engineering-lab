# ReAct with LangGraph

A ReAct agent built as an explicit LangGraph state graph: a reasoning node calls the LLM, a tool node runs the requested tools, and a conditional edge loops between them until the model stops calling tools.

**Skills:** `ReAct-agents` `state-graphs` `tool-calling` `conditional-edges`
**Stack:** `LangGraph` `LangChain` `Ollama` `Tavily`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled.

```bash
cp .env.example .env   # needs TAVILY_API_KEY; LangSmith keys are optional, for tracing
uv sync && uv run main.py
```
