from langchain_core.runnables import RunnableLambda


def extract_question(data):
    print("Step 1 input:", data)

    question = data["question"]

    print("Step 1 output:", question)
    return question


def uppercase_question(question):
    print("Step 2 input:", question)

    result = question.upper()

    print("Step 2 output:", result)
    return result


chain = (
    RunnableLambda(extract_question)
    | RunnableLambda(uppercase_question)
)

result = chain.invoke({
    "question": "What is RAG?"
})

print("\nFinal result:", result)