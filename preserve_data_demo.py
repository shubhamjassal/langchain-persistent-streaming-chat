
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def analyze_question(data):
    question = data["question"]

    return {
        "question_length": len(question),
        "question_uppercase": question.upper(),
    }


def create_summary(data):
    analysis = data["analysis"]

    return (
        f"Your question has {analysis['question_length']} characters. "
        f"Uppercase version: {analysis['question_uppercase']}"
    )


chain = (
    RunnablePassthrough.assign(
        analysis=RunnableLambda(analyze_question)
    )
    | RunnablePassthrough.assign(
        summary=RunnableLambda(create_summary)
    )
)

input_data = {
    "question": "What is RAG?",
    "student_level": "beginner",
}

result = chain.invoke(input_data)

print("Final output:")
print(result)
print("\nSummary:")
print(result["summary"])
