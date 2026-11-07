# Reflexion Agent

A research agent that answers, critiques its own answer, proposes search queries, runs them with Tavily and revises the answer with citations, repeating for a fixed number of iterations. Structured output comes from Pydantic schemas bound as tools.

**Skills:** `reflexion` `self-critique` `structured-output` `tool-calling` `web-search`
**Stack:** `LangGraph` `LangChain` `OpenAI` `Tavily` `Pydantic`

## Run

```bash
cp .env.example .env   # needs OPENAI_API_KEY and TAVILY_API_KEY; LangSmith keys are optional
uv sync && uv run main.py
```

Small local models struggle with the nested `AnswerQuestion` schema, so this project uses an OpenAI model.
