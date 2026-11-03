from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "granite4.2:3b"


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
    pass


def main() -> None:
    result = run_agent("What is the price of a laptop with a silver discount?")
