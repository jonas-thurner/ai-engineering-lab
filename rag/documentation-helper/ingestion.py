import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

# Load .env from this file's folder, regardless of the working directory
# (the debugger may start from elsewhere and miss it otherwise)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Use certifi's up-to-date CA bundle for HTTPS verification, so API calls
# (OpenAI, Tavily, Pinecone) don't fail with "CERTIFICATE_VERIFY_FAILED"
# on systems like macOS whose Python lacks root certificates.
# - ssl_context: SSL context built from certifi (currently unused)
# - SSL_CERT_FILE: default CA file for Python's ssl module / httpx
# - REQUESTS_CA_BUNDLE: CA file for the `requests` library
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    # Avoid Rate-Limiting Issues
    chunk_size=50,
    retry_min_seconds=10,
)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
# Extract: fetch the content of specific, known URLs (currently unused)
tavily_extract = TavilyExtract()
# Map: discover a site's URLs without fetching content (currently unused)
# - max_depth: how many link levels deep to follow from the start URL
# - max_breadth: max links to follow per page
# - max_pages: hard cap on total URLs discovered
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
# Crawl: map + extract in one step — follows links and returns page content
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
        except Exception as e:
            log_error(f"Batch {batch_num} failed: {type(e).__name__}: {e}")
            return False
        return True

    tasks = [add_batch(batch,i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True) == len(batches)

    if not successful:
        raise RuntimeError("At least one batch failed.")

async def main():
    """
    Main async function to orchestrate the entire process.
    """
    # Crawl the documentation site of langchain
    res = tavily_crawl.invoke(
        {
            "url": "https://sigmahq.io/",
            # Max is 5; best is starting by 1 and increasing
            "max_depth": 5,
            "extract_depth": "advanced",
            # An ai agent can decide wether a page is included using these instructions
            # "instructions": "content on ai agents"
        }
    )
    all_docs = [
        Document(page_content=r["raw_content"], metadata={"source": r["url"]})
        for r in res["results"]
        # Skip pages Tavily couldn't extract (raw_content is None)
        if r.get("raw_content")
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    # Avoid Rate-Limiting by the Vectorstore
    await index_documents_async(chunks, batch_size=500)


if __name__ == "__main__":
    asyncio.run(main())
