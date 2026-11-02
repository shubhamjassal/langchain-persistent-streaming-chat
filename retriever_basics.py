
from langchain_core.documents import Document

documents = [
    Document(
        page_content="RAG combines retrieval with generation "
                     "to answer questions using external information.",
        metadata={"topic": "RAG"},
    ),
    Document(
        page_content="Embeddings represent text as numerical vectors "
                     "that capture semantic meaning.",
        metadata={"topic": "Embeddings"},
    ),
    Document(
        page_content="LCEL connects LangChain components into "
                     "composable processing pipelines.",
        metadata={"topic": "LCEL"},
    ),
]

print("Total documents:", len(documents))

for document in documents:
    print("\nTopic:", document.metadata["topic"])
    print("Content:", document.page_content)