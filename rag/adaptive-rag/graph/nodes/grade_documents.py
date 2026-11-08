from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.consts import MIN_RELEVANT_DOCUMENTS
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    relevant_documents = [
        doc
        for doc in state.documents
        if retrieval_grader.invoke(
            {"question": state.question, "document": doc.page_content}
        ).is_relevant
    ]
    # Fall back to web search if too few relevant documents are left to answer the question
    web_search = len(relevant_documents) < MIN_RELEVANT_DOCUMENTS
    return {"documents": relevant_documents, "web_search": web_search}
