from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from graph.chains.answer_grader import GradeAnswer, answer_chain
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
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


def test_hallucination_grader_answer_yes() -> None:
    # Answer generated from the retrieved docs themselves -> is_grounded is True
    question = "agent memory"
    docs = retriever.invoke(question)
    documents = "\n\n".join(doc.page_content for doc in docs)
    generation: str = generation_chain.invoke(
        {"context": documents, "question": question}
    )

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    assert res.is_grounded


def test_hallucination_grader_answer_no() -> None:
    # Made-up answer with no support in the retrieved docs -> is_grounded is False
    question = "agent memory"
    docs = retriever.invoke(question)
    documents = "\n\n".join(doc.page_content for doc in docs)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": "To make a good pizza, bake the dough at 250 degrees Celsius for ten minutes.",
        }
    )
    assert not res.is_grounded


def test_answer_grader_answer_yes() -> None:
    # Answer that directly addresses the question -> is_useful is True
    question = "What is short-term memory in LLM agents?"
    generation = "Short-term memory in LLM agents is in-context learning: the information kept in the prompt within the model's context window."

    res: GradeAnswer = answer_chain.invoke(
        {"question": question, "generation": generation}
    )
    assert res.is_useful


def test_answer_grader_answer_no() -> None:
    # Off-topic answer that ignores the question -> is_useful is False
    question = "What is short-term memory in LLM agents?"
    generation = (
        "To make a good pizza, bake the dough at 250 degrees Celsius for ten minutes."
    )

    res: GradeAnswer = answer_chain.invoke(
        {"question": question, "generation": generation}
    )
    assert not res.is_useful
