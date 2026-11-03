from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()


# The tool decorator uses the function name as the tool name,
# the function parameters as the tool input schema,
# and the function docstring as the tool description.
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatOllama(model="qwen3.8")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke({"messages": [
        HumanMessage(content="What is the weather in Tokyo?")
        ]})
    print(result)


if __name__ == "__main__":
    main()
