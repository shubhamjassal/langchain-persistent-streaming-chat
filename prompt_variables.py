
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    (
        "human",
        "Explain {topic} for a {level} learner "
        "in {points} bullet points."
    ),
])

user_input = {
    "topic": "RAG",
    "level": "beginner",
}

# Add a default if "points" was not provided
user_input.setdefault("points", 3)

print("Input sent to prompt:", user_input)

messages = prompt.invoke(user_input)

print("\nFormatted human message:")
print(messages.messages[1].content)
