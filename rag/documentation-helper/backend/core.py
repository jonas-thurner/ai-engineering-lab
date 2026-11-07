import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
# Initialize chat model
model = init_chat_model("ollama:granite4.2:3b")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant documentation to help answer user queries about SigmaHQ.
    """
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unkown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    """

    system_prompt = """
    You are an assistant for the SigmaHQ documentation and Sigma detection rules.
    You have access to a tool that retrieves documentation.
    Use the tool before answering. 
    Cite sources in your answers. 
    If you cannot find the answer in the retrieved documentation, say so.
    """

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [HumanMessage(content=query)]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    # Flatten the retrieved docs from all tool calls into one list
    context_docs = [
        doc
        for m in response["messages"]
        if isinstance(m, ToolMessage) and isinstance(m.artifact, list)
        for doc in m.artifact
    ]

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="What is a SigmaHQ detection rule?")
    print(result)
