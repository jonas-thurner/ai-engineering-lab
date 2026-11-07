from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# llm = init_chat_model("openai:gpt-6-luna")
llm = init_chat_model("ollama:granite4.2:3b")

SYSTEM = """
    You are an assistant for question-answering tasks.
    Use only the retrieved context below to answer the question.
    If the context does not contain the answer, say that you don't know. Do not make anything up.
    Keep the answer concise: three sentences maximum.
    End with the source URLs you used.
"""

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Question: {question} \n\n Context: \n\n {context}"),
    ]
)

generation_chain = generation_prompt | llm | StrOutputParser()
