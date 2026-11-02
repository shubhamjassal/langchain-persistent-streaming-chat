
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)

def prepare_topic(data):
    return f"Please explain: {data['topic']}"

def fake_model(data):
    return f"Here is a simple explanation of {data['prepared']}"

chain = (
    RunnableParallel(
        original=RunnablePassthrough(),
        prepared=RunnableLambda(prepare_topic),
    )
    | RunnablePassthrough.assign(
        answer=RunnableLambda(fake_model)
    )
)

result = chain.invoke({"topic": "RAG"})

print("Original input:", result["original"])
print("Prepared prompt:", result["prepared"])
print("Final answer:", result["answer"])
