import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

summary_template = """
# Instruction

Given the following information about a person.

I want you to create: 
1. A short summary in 2-3 sentences.
2. Two intersting facts about the person.

## Information

{info}
"""


def main():
    print("Hello from langchain-course!")

    loader = TextLoader("elonmusk.info.txt", encoding="utf-8")
    documents = loader.load()
    musk_info = documents[0].page_content

    prompt_template = PromptTemplate(
        input_variables=["info"],
        template=summary_template,
    )

    llm = ChatOpenAI(model_name="gpt-6-luna")

    # A runnable object is created by combining the prompt template and the LLM.
    # The prompt template is automatically mapped to the input of the LLM.
    chain = prompt_template | llm

    response = chain.invoke({"info": musk_info})
    print(response.content)


if __name__ == "__main__":
    main()
