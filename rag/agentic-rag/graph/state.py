from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """
    Represents the state of our graph.
    """

    question: str = Field(description="The user's question.")
    response: str = Field(default="", description="The LLM's generated answer.")
    web_search: bool = Field(default=False, description="Whether to run a web search.")
    documents: List[Document] = Field(
        default_factory=list,
        description="Retrieved documents used to answer the question.",
    )
