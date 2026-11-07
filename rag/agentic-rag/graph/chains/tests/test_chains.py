from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    # Retrieve docs for a question, grade one of them against the same question -> is_relevant is True
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.is_relevant


def test_retrieval_grader_answer_no() -> None:
    # Off-topic question: the retriever still returns its closest chunk, the grader must reject it -> is_relevant is False
    question = "Who is harry kane?"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert not res.is_relevant


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    # Pass only the text: the Document metadata alone overflows the model's context window
    context = "\n\n".join(doc.page_content for doc in docs)

    generation: str = generation_chain.invoke(
        {"context": context, "question": question}
    )
    pprint(generation)
