from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from chains import generate_chain, reflection_chain

load_dotenv()

GENERATE = "GENERATE"
REFLECT = "REFLECT"


def generation_node(state: MessagesState):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessagesState):
    res = reflection_chain.invoke({"messages": state["messages"]})
    # Send the critique back as a HumanMessage
    # so the LLM treats it as human user feedback
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessagesState):
    if len(state["messages"]) > 5:
        return END
    return REFLECT


flow = StateGraph(MessagesState)
flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)

flow.set_entry_point(GENERATE)
flow.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
flow.add_edge(REFLECT, GENERATE)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


def main():
    print("Hello from reflection-agent!")


if __name__ == "__main__":
    main()
