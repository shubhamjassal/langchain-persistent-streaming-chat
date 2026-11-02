
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

    for document, score in results:
        print(f"Score: {score:.3f}")
        print(f"Document: {document.page_content}")
