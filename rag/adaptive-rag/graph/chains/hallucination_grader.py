from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

# temperature=0: grading should be deterministic, not creative
llm = init_chat_model("ollama:granite4.2:3b", temperature=0)


class GradeHallucinations(BaseModel):
    is_grounded: bool = Field(
        description="True if the answer is grounded in the facts. False if the answer is not grounded in the facts."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

SYSTEM = """
You check whether an answer is supported by the given facts.
If the answer talks about anything the facts do not mention, it is NOT grounded (False).
If the answer talks refelects the given facts, the answer is grounded (True).
"""

# Small models lose the task after long context, so the question comes last, right after the answer
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            "Facts:\n\n{documents}\n\n---\n\nAnswer:\n{generation}\n\n"
            "Task: Is the answer grounded in the facts above?",
        ),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
