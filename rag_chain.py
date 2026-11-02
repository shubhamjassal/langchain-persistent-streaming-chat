
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()

# 1. Create sample documents
# 1. Create sample documents
documents = [
    Document(
        page_content=(
            "Embeddings convert text into numerical vectors. "
            "These vectors represent aspects of the text's meaning. "
            "Texts with similar meanings can have vectors that are "
            "close to each other in the embedding space."
        ),
        metadata={"topic": "Embeddings"},
    ),
    Document(
        page_content=(
            "Semantic search finds information based on meaning "
            "rather than requiring exact keyword matches. "
            "It converts a user's query into an embedding, compares "
            "that vector with document embeddings, and retrieves "
            "documents with high similarity."
        ),
        metadata={"topic": "Semantic Search"},
    ),
    Document(
        page_content=(
            "RAG stands for Retrieval-Augmented Generation. "
            "A RAG system retrieves relevant documents and supplies "
            "their content as context to a language model. "
            "The language model uses this context to generate an answer."
        ),
        metadata={"topic": "RAG"},
    ),
    Document(
        page_content=(
            "LCEL stands for LangChain Expression Language. "
            "It connects components such as retrievers, prompts, "
            "language models, and output parsers into processing chains."
        ),
        metadata={"topic": "LCEL"},
    ),
]
# 2. Create embeddings and index documents
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)

# 3. Retrieve only documents above the threshold
THRESHOLD = 0.60

def retrieve_relevant_docs(inputs):
    results = vector_store.similarity_search_with_score(
        inputs["question"], k=2
    )

    print("\nRetrieval diagnostics:")

    for doc, score in results:
        status = "ACCEPTED" if score >= THRESHOLD else "REJECTED"

        print(
            f"{doc.metadata['topic']}: "
            f"score={score:.4f} | {status}"
        )

    return [
        doc for doc, score in results
        if score >= THRESHOLD
    ]

# 4. Format retrieved documents
def format_documents(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# 5. Create the prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question:
{question}
""")

# 6. Configure Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
output_parser = StrOutputParser()

# 7. Build one chain that returns documents and answer
retrieval_chain = RunnablePassthrough.assign(
    docs=RunnableLambda(retrieve_relevant_docs)
).assign(
    context=lambda inputs: format_documents(inputs["docs"])
)

answer_chain = (
    {
        "context": lambda inputs: inputs["context"],
        "question": lambda inputs: inputs["question"],
    }
    | prompt
    | model
    | output_parser
)

def generate_answer(inputs):
    if not inputs["docs"]:
        return "I don't know."

    return answer_chain.invoke(inputs)


full_chain = retrieval_chain | RunnablePassthrough.assign(
    answer=RunnableLambda(generate_answer)
)


# 8. Ask a question
question = input("Ask a question: ").strip()

if question:
    result = full_chain.invoke({"question": question})

    print(f"\nRelevance threshold: {THRESHOLD:.2f}")
    print("\nRetrieved documents:\n")

    if result["docs"]:
        for index, doc in enumerate(result["docs"], start=1):
            print(f"Document {index}: {doc.metadata['topic']}")
            print(doc.page_content)
            print()
    else:
        print("No sufficiently similar documents found.")

    print("Answer:\n")
    print(result["answer"])
else:
    print("Please enter a question.")
