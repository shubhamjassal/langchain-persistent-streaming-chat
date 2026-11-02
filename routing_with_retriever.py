
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Sample documents for our first retrieval test
documents = [
    "RAG stands for Retrieval-Augmented Generation. "
    "It retrieves relevant information and gives it to an LLM as context.",

    "Python is a programming language used for automation, "
    "web development, data science, and artificial intelligence.",

    "ChromaDB is a vector database that can store embeddings "
    "and retrieve semantically similar documents.",
]


# Temporary retriever: select documents using keyword overlap.
# This is a simple testing substitute, not semantic search.
def retrieve_documents(data):
    question_words = set(data["question"].lower().split())

    matches = [
        doc for doc in documents
        if question_words.intersection(set(doc.lower().split()))
    ]

    return {
        **data,
        "context": "\n".join(matches) if matches else "No matching document found."
    }


# Answer using the retrieved context
document_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context does not contain the answer, say you don't have enough information.

Context:
{context}

Question:
{question}
"""
)

document_chain = (
    RunnableLambda(retrieve_documents)
    | document_prompt
    | model
)

# General-question path
general_prompt = ChatPromptTemplate.from_template(
    "Answer this question clearly and concisely:\n{question}"
)

general_chain = general_prompt | model

# Conditional router
router = RunnableBranch(
    (
        lambda data: data["type"] == "document",
        document_chain,
    ),
    general_chain,
)

questions = [
    {
        "type": "document",
        "question": "What does RAG stand for?",
    },
    {
        "type": "general",
        "question": "What is Python?",
    },
]

for question in questions:
    print(f"\nQuestion: {question['question']}")
    result = router.invoke(question)
    print("Response:", result.text)
