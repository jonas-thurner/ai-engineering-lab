# Documentation Helper

Chat assistant over the [SigmaHQ](https://sigmahq.io/) documentation that answers detection-rule questions and cites its sources.

**Skills:** `RAG` `web-crawling` `agentic-retrieval` `chat-UI`
**Stack:** `LangChain` `Tavily` `Pinecone` `OpenAI` `Ollama` `Streamlit`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled and a Pinecone index.

```bash
cp .env.example .env
pipenv install
pipenv run python ingestion.py      # crawl and index the docs once
pipenv run streamlit run main.py
```

---
Based on [emarco177/documentation-helper](https://github.com/emarco177/documentation-helper) (Apache-2.0).
