# RAG Gist

Retrieval-augmented generation over the NIST Cybersecurity Framework PDF, with the retrieval step wired up by hand.

**Skills:** `RAG` `chunking` `embeddings` `vector-search`
**Stack:** `LangChain` `Pinecone` `OpenAI` `Ollama` `Unstructured`

## Run

Requires [Ollama](https://ollama.com) with `granite4.2:3b` pulled and a Pinecone index.

```bash
cp .env.example .env
uv sync
uv run python -c "import main; main.init_pinecone()"   # ingest once
uv run main.py
```
