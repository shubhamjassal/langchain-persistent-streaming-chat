
from langchain_core.runnables import RunnableLambda

def explain_topic(data):
    topic = data["topic"]
    return f"Let's learn {topic} step by step."

model = RunnableLambda(explain_topic)

for chunk in model.stream({"topic": "RAG"}):
    print(chunk)
