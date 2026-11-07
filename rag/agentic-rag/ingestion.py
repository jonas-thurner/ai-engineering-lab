from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

COLLECTION_NAME = "rag-chroma"
PERSIST_DIRECTORY = "./.chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://www.anthropic.com/engineering/building-effective-agents",
    "https://www.anthropic.com/news/contextual-retrieval",
    "https://blog.langchain.com/agentic-rag-with-langgraph/",
    "https://eugeneyan.com/writing/llm-patterns/",
]


def ingest():
    """Load the URLs, split them into chunks and store them in Chroma."""
    docs = [
        UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000
        ).load()
        for url in urls
    ]
    # Flatten the document list.
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )


# Only reads the existing store, so importing this module never re-ingests
retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


if __name__ == "__main__":
    ingest()
