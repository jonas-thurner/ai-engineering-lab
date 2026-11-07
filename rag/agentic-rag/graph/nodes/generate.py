from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    # Pass only the text plus its URL: the full Document metadata overflows the model's context window
    context = "\n\n".join(
        f"{doc.page_content}\nSource: {doc.metadata['url']}" for doc in state.documents
    )

    res = generation_chain.invoke({"context": context, "question": state.question})
    return {"response": res}
