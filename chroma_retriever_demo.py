
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Create sample documents
documents = [
    Document(
        page_content=(
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information and provides it "
            "to a language model as context."
        ),
        metadata={"topic": "RAG"},
    ),
    Document(
        page_content=(
            "Python is a programming language used for automation, "
            "web development, data science, and artificial intelligence."
        ),
        metadata={"topic": "Python"},
    ),
    Document(
        page_content=(
            "ChromaDB stores vector embeddings and supports "
            "similarity search over documents."
        ),
        metadata={"topic": "ChromaDB"},
    ),
]

# 2. Configure Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 3. Create a persistent ChromaDB vector store
vector_store = Chroma(
    collection_name="routing_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_routing_db",
)
# Remove records from the initial run that used auto-generated IDs
vector_store.delete(
    ids=[
        "749ce3d8-421f-4cef-938c-9a56ce7e40a6",
        "6f2d19be-20c5-481d-b7ac-dec088389f4d",
        "7e90f732-50bd-4d1f-a1d5-bfc6c6830202",
    ]
)

# Add or update documents using stable IDs
vector_store.add_documents(
    documents,
    ids=["rag-guide", "python-guide", "chroma-guide"],
)
# 4. Add the sample documents

print("Total documents in database:", vector_store._collection.count())

stored_data = vector_store.get(include=["documents", "metadatas"])
for doc_id, content in zip(stored_data["ids"], stored_data["documents"]):
    print(f"\nID: {doc_id}")
    print(f"Content: {content}")
# 5. Retrieve relevant documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

question = "What does RAG mean?"
results = retriever.invoke(question)

print("\nRetrieved documents:")
for index, document in enumerate(results, start=1):
    print(f"\n{index}. {document.page_content}")
    print("Metadata:", document.metadata)
