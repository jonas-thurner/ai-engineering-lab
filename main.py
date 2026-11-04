import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def main():
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

if __name__ == "__main__":
    main()
