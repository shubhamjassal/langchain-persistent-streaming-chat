
from langchain_core.runnables import RunnableBranch, RunnableLambda


def answer_general(data):
    return f"Direct-answer path: {data['question']}"


def answer_from_documents(data):
    return f"Document-retrieval path: {data['question']}"


router = RunnableBranch(
    (
        lambda data: data["type"] == "document",
        RunnableLambda(answer_from_documents),
    ),
    RunnableLambda(answer_general),  # Default path
)


questions = [
    {
        "type": "general",
        "question": "What is Python?",
    },
    {
        "type": "document",
        "question": "What does my PDF say about RAG?",
    },
    {
    "type": "unknown",
    "question": "How do AI models learn?",
},
]

for question in questions:
    result = router.invoke(question)
    print(result)
