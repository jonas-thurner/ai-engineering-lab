from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


def main():
    question = "What is agent memory?"
    res = app.invoke({"question": question})
    print(res["response"])


if __name__ == "__main__":
    main()
