from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tavily import TavilyClient

tavily = TavilyClient()


class Source(BaseModel):
    """
    Schema for a source used by the agent.
    """

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """
    Schema for the agent's response.
    """

    answer: str = Field(description="The answer provided by the agent.")
    # default_factory=list is used to create a new empty list for each instance of AgentResponse.
    sources: list[Source] = Field(
        default_factory=list, description="A list of sources used by the agent."
    )


# The tool decorator uses the function name as the tool name,
# the function parameters as the tool input schema,
# and the function docstring as the tool description.
@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet.
    Args:
        query (str): The search query.
    Returns:
        dict: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatOllama(model="qwen3.8")
tools = [search]

# With Ollama, create_agent() offers AgentResponse to the model as an extra "tool".
# Ollama can't force the model to call it, so without this instruction the model
# often just answers in plain text and structured_response stays None.
system_prompt = (
    "You are a research assistant. Use the search tool to find information. "
    "When you are done, you MUST give your final answer by calling the AgentResponse tool, "
    "including the URLs of the sources you used. Never answer in plain text."
)
agent = create_agent(
    model=llm, tools=tools, system_prompt=system_prompt, response_format=AgentResponse
)


def main():
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    # agent.invoke() returns the agent's final state as a dict:
    #   result["messages"]: the full conversation, in order:
    #     HumanMessage  -> our question
    #     AIMessage     -> the model decides to call a tool (tool_calls=[search(...)])
    #     ToolMessage   -> what our search tool returned
    #     AIMessage     -> the model's final answer (repeats if more tools are needed)
    # result["structured_response"]: only present because we passed response_format=AgentResponse to create_agent().
    structured = result.get("structured_response")
    if structured is None:
        # The model ignored the format and answered in plain text.
        print(f"(no structured response) {result['messages'][-1].content}")
        return

    print(f"Answer: {structured.answer}")
    print(f"Sources: {structured.sources}")


if __name__ == "__main__":
    main()
