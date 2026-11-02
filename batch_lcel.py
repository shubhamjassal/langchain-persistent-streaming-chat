
from langchain_core.runnables import RunnableLambda

# A simple fake model
def explain_topic(data):
    topic = data["topic"]
    return f"Learning topic: {topic}"

model = RunnableLambda(explain_topic)

# Run the same component with three inputs
results = model.batch([
    {"topic": "RAG"},
    {"topic": "embeddings"},
    {"topic": "vector databases"},
])

for result in results:
    print(result)
