import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print(os.environ.get("OPENAI_API_KEY"))
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
