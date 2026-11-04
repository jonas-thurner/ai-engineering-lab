import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

retriever = None
llm = None

def init_pinecone():
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="nist_csf.pdf",
        chunking_strategy="basic",
        max_characters=1000000,
        include_orig_elements=False,
    )
    document = loader.load()

    print("Chunking...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    texts = text_splitter.split_documents(document)

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("Creating vector store...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"],
    )

    print("Finished!")

def get_retriever():
    global retriever
    if retriever is not None:
        return retriever
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def get_llm():
    global llm
    if llm is not None:
        return llm
    llm = ChatOllama(
        model="granite4.2:3b"
    )
    return llm

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context.

    {context}

    Question: {question}
    """
)

def format_docs(docs):
    "Format docs retrieved from Pinecone to be used in the prompt template."
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_langchain(query: str):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    formatted_docs = format_docs(docs)
    prompt = prompt_template.format(
        context=formatted_docs,
        question=query
    )
    message = HumanMessage(content=prompt)
    response = get_llm().invoke([message])
    return response.content




def main():
    question = "What are Roles, Responsibilities, and Authorities (GV.RR)?"
    print(retrieval_chain_without_langchain(question))
    # retriever = get_retriever()
    # llm = get_llm()
    # 

    # response = llm.invoke(
    #     messages=[
    #         HumanMessage(content=question)
    #     ]
    # )

    # print(response.content)


if __name__ == "__main__":
    main()
