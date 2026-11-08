from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# llm = init_chat_model("openai:gpt-6-luna")
llm = init_chat_model("ollama:granite4.2:3b")


class GradeDocuments(BaseModel):
    is_relevant: bool = Field(
        description="True if the document is relevant to the question, otherwise False."
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYSTEM = """
    You are a grader assessing relevance of a retrieved document to a user question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
