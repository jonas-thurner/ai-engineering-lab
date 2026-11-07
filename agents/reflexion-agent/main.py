from typing import Literal

from chains import first_responder, revisor
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from tool_executor import execute_tools
from langsmith import traceable

MAX_ITERATIONS = 2

DRAFT = "draft"
EXECUTE_TOOLS = "execute_tools"
REVISE = "revise"


def draft_node(state: MessagesState):
    res = first_responder.invoke(input={"messages": state["messages"]})
    return {"messages": [res]}


def revise_node(state: MessagesState):
    res = revisor.invoke(input={"messages": state["messages"]})
    return {"messages": [res]}


def event_loop(state: MessagesState) -> Literal["execute_tools", "__end__"]:
    # Every search round adds ToolMessages, so their count tells us how many rounds have run
    num_iterations = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS


flow = StateGraph(MessagesState)
flow.add_node(DRAFT, draft_node)
flow.add_node(EXECUTE_TOOLS, execute_tools)
flow.add_node(REVISE, revise_node)

flow.set_entry_point(DRAFT)
flow.add_edge(DRAFT, EXECUTE_TOOLS)
flow.add_edge(EXECUTE_TOOLS, REVISE)
flow.add_conditional_edges(REVISE, event_loop, {EXECUTE_TOOLS: EXECUTE_TOOLS, END: END})

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

@traceable(name="Reflexion: AI SOC research")
def main():
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
                )
            ]
        }
    )
    print(res["messages"][-1].tool_calls[0]["args"]["answer"])


if __name__ == "__main__":
    main()
