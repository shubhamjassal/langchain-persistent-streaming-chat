
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    collection_name="routing_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_routing_db",
)

THRESHOLD = 0.50

questions = [
    "What does RAG stand for?",
    "What is ChromaDB used for?",
    "What is the capital of France?",
]

for question in questions:
    print(f"\nQuestion: {question}")

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=3,
    )

    accepted = [
        (document, score)
        for document, score in results
        if score >= THRESHOLD
    ]

    if accepted:
        print("Relevant documents found:")
        for document, score in accepted:
            print(f"  Score: {score:.3f} | {document.page_content}")
    else:
        print("No sufficiently relevant documents found.")
