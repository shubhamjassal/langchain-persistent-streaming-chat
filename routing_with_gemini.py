
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

# Path 1: Answer general questions with Gemini
general_prompt = ChatPromptTemplate.from_template(
    "Answer this question clearly and simply:\n{question}"
)

general_chain = general_prompt | model

# Path 2: Document questions need retrieved document context
def document_response(data):
    return (
        "This question requires document retrieval before it can be "
        f"answered reliably. Question: {data['question']}"
    )

document_chain = RunnableLambda(document_response)

# Route each input to the appropriate path
router = RunnableBranch(
    (
        lambda data: data["type"] == "document",
        document_chain,
    ),
    general_chain,  # Default path
)

questions = [
    {
        "type": "general",
        "question": "What is Python? Explain in two sentences.",
    },
    {
        "type": "document",
        "question": "What does my PDF say about RAG?",
    },
]

for question in questions:
    print(f"\nQuestion: {question['question']}")
    result = router.invoke(question)
    if hasattr(result, "content"):
        response = result.text
    else:
        response = result

    print("Response:", response)
