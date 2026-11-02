
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
load_dotenv()
from routing_logic import build_router

from langchain_core.exceptions import ModelAPIError, ModelConnectionError, ModelTimeoutError
from langchain_google_genai.chat_models import GoogleRateLimitError

import os

# 1. Initialize Gemini
model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    # keep your other existing settings
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 2. Load the existing persistent ChromaDB collection
vector_store = Chroma(
    collection_name="routing_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_routing_db",
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# 3. General-question path
general_prompt = ChatPromptTemplate.from_template(
    "Answer this question clearly and concisely:\n{question}"
)

def normalize_general_response(message):
    return {
        "answer": message.text,
        "source": "gemini",
        "status": "success",
    }



retrying_model = model.with_retry(
    retry_if_exception_type=(
        GoogleRateLimitError,
        ModelAPIError,
        ModelConnectionError,
        ModelTimeoutError,
    ),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

general_chain = (
    general_prompt
    | retrying_model
    | RunnableLambda(normalize_general_response)
)

# 4. Document-question path

def retrieve_context(data):
    threshold = 0.50

    results = vector_store.similarity_search_with_relevance_scores(
    data["question"],
    k=3,
)
    relevant_docs = [
        doc
        for doc, score in results
        if score >= threshold
    ]
    for doc in relevant_docs:
        print("Document metadata:", doc.metadata)

    if not relevant_docs:
        return {
            "question": data["question"],
            "context": "",
            "rejected": True,
        }

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    stored_data = vector_store.get(
    include=["documents", "metadatas"]
)

    sources = []

    for doc in relevant_docs:
        for doc_id, content in zip(
        stored_data["ids"],
        stored_data["documents"],
    ):
          if content == doc.page_content:
            sources.append(doc_id)
            break

    return {
        "question": data["question"],
        "context": context,
        "sources": sources,
        "rejected": False,
    }



document_prompt = ChatPromptTemplate.from_template(
    """Answer using only the provided context.
If the answer is not in the context, say you don't have enough information.

Context:
{context}

Question:
{question}
"""
)

def normalize_document_response(data):
    return {
        "answer": data["answer"].text,
        "source": "chroma",
        "status": "success",
        "sources": data["sources"],
    }


def rejection_response(data):
    return {
        "answer": (
            "I couldn't find sufficiently relevant information "
            "in the available documents to answer this question."
        ),
        "source": "chroma",
        "status": "no_relevant_context",
        "sources": [],
    }



document_chain = (
    RunnableLambda(retrieve_context)
    | RunnableBranch(
        (
            lambda data: data["rejected"],
            RunnableLambda(rejection_response),
        ),
        (
            RunnableLambda(
                lambda data: {
                    "question": data["question"],
                    "context": data["context"],
                    "sources": data["sources"],
                }
            )
            | RunnablePassthrough.assign(
                answer=document_prompt | model
            )
            | RunnableLambda(normalize_document_response)
        ),
    )
)


# 5. Build the question router


router = build_router(document_chain, general_chain)

# 6. Test both routes

def process_questions(questions, router):
    results = []

    for question in questions:
        print(f"\nQuestion: {question['question']}")

        try:
            result = router.invoke(question)

        except Exception as error:
            result = {
        "answer": "Unable to process this question right now.",
        "source": "none",
        "status": "error",
        "sources": [],
        "error": str(error),
           }
            print(f"Error: {error}")

        results.append(result)

        print("Answer:", result["answer"])
        print("Source:", result["source"])
        print("Status:", result["status"])
        print("Sources:", result.get("sources", []))

    return results

if __name__ == "__main__":
    # Keep your existing questions list here.
    # Keep your existing for-loop here
    questions = [
    {
        "type": "document",
        "question": "What does RAG stand for?",
    },
    {
        "type": "document",
        "question": "What is the capital of France?",
    },
    {
        "type": "general",
        "question": "Why is Python popular for AI development?",
    },
]