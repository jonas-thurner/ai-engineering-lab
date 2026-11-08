from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_chain
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import (GENERATE, GRADE_DOCUMENTS, NOT_SUPPORTED, NOT_USEFUL,
                          RETRIEVE, USEFUL, WEB_SEARCH)
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


def decide_to_generate(state: GraphState) -> Literal["web_search", "generate"]:
    if state.web_search:
        return WEB_SEARCH
    return GENERATE


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not supported"]:
    documents = "\n\n".join(doc.page_content for doc in state.documents)

    hallucination_grade = hallucination_grader.invoke(
        {"documents": documents, "generation": state.response}
    )
    if not hallucination_grade.is_grounded:
        return NOT_SUPPORTED

    answer_grade = answer_chain.invoke(
        {"question": state.question, "generation": state.response}
    )
    if answer_grade.is_useful:
        return USEFUL
    return NOT_USEFUL


flow = StateGraph(GraphState)
flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(WEB_SEARCH, web_search)
flow.add_node(GENERATE, generate)

flow.set_entry_point(RETRIEVE)
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)
flow.add_edge(WEB_SEARCH, GENERATE)
flow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {NOT_USEFUL: WEB_SEARCH, NOT_SUPPORTED: GENERATE, USEFUL: END},
)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
