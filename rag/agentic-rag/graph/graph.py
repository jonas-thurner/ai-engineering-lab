from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from langgraph.graph import END, StateGraph


def decide_to_generate(state: GraphState) -> Literal["web_search", "generate"]:
    if state.web_search:
        return WEB_SEARCH
    return GENERATE


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
flow.add_edge(GENERATE, END)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
