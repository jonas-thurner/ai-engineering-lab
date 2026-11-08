from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    is_useful: bool = Field(
        description="If the answer is useful return True, if not return False."
    )


llm = init_chat_model("ollama:granite4.2:3b", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

SYSTEM = """
You check whether an answer actually answers the user's question.
If the answer addresses the question directly, it is useful.
If it is off-topic, says it doesn't know, or misses the point, it is NOT useful.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            "Question: {question}\n\nAnswer:\n{generation}\n\nDoes the answer address the question?",
        ),
    ]
)

answer_chain: RunnableSequence = answer_prompt | structured_llm_grader
