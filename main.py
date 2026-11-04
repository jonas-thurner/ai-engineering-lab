from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "granite4.2:3b"

SYSTEM_PROMPT = """
You are a helpful shopping assistant.
You have access to a product price and discount tool.

STRICT RULES: 
You must follow these exactly: 
1. NEVER guess or assume any product price.
You musst call get_product_price to get the price of a product.
2. NEVER guess or assume any discount.
You must call apply_discount to get the discounted price of a product.
3. ALWAYS use the tools provided to get the price and discount. 
NEVER make up a price or use math to calculate a discount. 
4. If a user does not provide a discount tier, you must ask them.
Do NEVER assume a discount tier.
"""


@tool
def get_product_price(product: str) -> float:
    """
    Tool that gets the price of a product.
    Args:
        product (str): The name of the product.
    Returns:
        float: The price of the product.
    """
    print(f"Getting price for: {product}")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    A tool that applies a discount to a price based on the discount tier.

    Args:
        price (float): The original price of the product.
        discount_tier (str): The discount tier, which can be "bronze", "silver", or "gold".

    Returns:
        float: The discounted price.
    """
    discount_factor = {
        "bronze": 0.95,  # 5% discount
        "silver": 0.90,  # 10% discount
        "gold": 0.80,  # 20% discount
    }
    return round(price * discount_factor.get(discount_tier, 1), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 50)

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}")

        # Thought or Answere from the AI
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            # Final Answere from the AI
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} ({tool_call_id}) with args: {tool_args}")

        tool = tools_dict.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found.")

        observation = tool.invoke(tool_args)

        # Keep the full history of AI messages
        messages.append(ai_message)
        # Add the observation from the tool
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )


def main() -> None:
    result = run_agent("What is the price of a laptop with a silver discount?")


if __name__ == "__main__":
    main()
