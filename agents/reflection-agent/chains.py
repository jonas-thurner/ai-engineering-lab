from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ChatPromptTemplate: a reusable chat prompt that becomes a list of messages once its variables are filled in
# from_messages builds it from (role, content) tuples and placeholders
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # Fixed system message: sets the model's role for every call
        (
            "system",
            "You are a viral LinkedIn influencer grading a post. "
            "Generate a critique and recommendations for the user's submission. "
            "Always provide detailed recommendations, including requests for length, virality, depth, style, etc.",
        ),
        # Injects the conversation history (submission and previous revisions) at this point
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Same structure, different system role: this one writes the post instead of grading it
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a LinkedIn content assistant tasked with writing excellent LinkedIn posts. "
            "Generate the best LinkedIn post possible for the user's request. "
            "If the user provides a critique, respond with a revised version of your previous attempt.",
        ),
        # Injects the request plus any earlier drafts and critiques
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = init_chat_model("ollama:granite4.2:3b")
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
