# Agentic RAG

Corrective RAG with LangGraph: retrieve chunks from a Chroma store, let an LLM grade each one for relevance, fall back to a Tavily web search when too few relevant chunks remain, and generate a concise answer that cites its source URLs.

**Skills:** `agentic-RAG` `corrective-RAG` `document-grading` `web-search-fallback` `source-citations` `pytest`
**Stack:** `LangGraph` `LangChain` `Chroma` `Tavily` `Ollama` `OpenAI` `Pydantic` `Unstructured`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled.

```bash
cp .env.example .env      # needs OPENAI_API_KEY (embeddings) and TAVILY_API_KEY
uv sync
uv run ingestion.py       # once: loads the blog posts into ./.chroma
uv run main.py
uv run pytest -s          # chain tests against the local store
```

Delete `.chroma` before re-running the ingestion, otherwise every chunk is stored twice.
