from typing import Any, Dict

from graph.consts import MAX_SEARCH_RESULTS
from graph.state import GraphState
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=MAX_SEARCH_RESULTS)


def web_search(state: GraphState) -> Dict[str, Any]:
    res = tavily.invoke({"query": state.question})["results"]
    docs = [
        Document(page_content=r["content"], metadata={"url": r["url"]}) for r in res
    ]
    # Keep the documents that passed the grader and add the web results
    return {"documents": state.documents + docs}
