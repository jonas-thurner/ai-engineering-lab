from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.consts import RETRIEVE, WEB_SEARCH


class RouteQuery(BaseModel):
    datasource: Literal["retrieve", "web_search"] = Field(
        description=f"Given a user question choose to route it to {WEB_SEARCH} or {RETRIEVE}."
    )


llm = init_chat_model("ollama:granite4.2:3b", temperature=0)

structured_llm_router = llm.with_structured_output(RouteQuery)

SYSTEM = f"""
You route a user question to {RETRIEVE} or to {WEB_SEARCH}.
{RETRIEVE} searches documents about LLM agents, prompt engineering, adversarial attacks on LLMs,
LLM hallucination, building effective agents, contextual retrieval, agentic RAG and LLM design patterns.
Use {RETRIEVE} for questions on these topics. Otherwise, use {WEB_SEARCH}.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "{question}"),
    ]
)

router_chain: RunnableSequence = route_prompt | structured_llm_router
