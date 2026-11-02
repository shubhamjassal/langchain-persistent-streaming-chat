
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Create sample documents
documents = [
    Document(
        page_content="RAG retrieves relevant information and uses an LLM to generate answers.",
        metadata={"topic": "RAG"},
    ),
    Document(
        page_content="Embeddings represent text as numerical vectors that capture meaning.",
        metadata={"topic": "Embeddings"},
    ),
    Document(
        page_content="LCEL connects LangChain components into processing pipelines.",
        metadata={"topic": "LCEL"},
    ),
]

# 2. Create an embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 3. Store the documents in a vector store
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)

# 4. Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# 5. Search for relevant documents
question = "How can AI find information by meaning instead of exact keywords?"
results = retriever.invoke(question)

print("Question:", question)
print("\nRetrieved documents:")

for document in results:
    print(f"\nTopic: {document.metadata['topic']}")
    print("Content:", document.page_content)
