from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
FETCH_COUNT = 8
COMPRESSED_COUNT = 3

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def create_retriever():
    """Create hybrid BM25/vector retrieval with contextual compression."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    stored_data = vector_store.get(include=["documents", "metadatas"])

    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            stored_data["documents"], stored_data["metadatas"]
        )
        if content
    ]
    if not documents:
        raise RuntimeError("No documents found. Run ingestion.py first.")

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = FETCH_COUNT
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FETCH_COUNT},
    )
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    compressor = DocumentCompressorPipeline(
        transformers=[
            EmbeddingsFilter(embeddings=embeddings, k=COMPRESSED_COUNT),
            LLMChainExtractor.from_llm(llm),
        ]
    )
    return ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=compressor,
    )


retriever = create_retriever()

prompt = ChatPromptTemplate.from_template(
    """Answer using only the information in the context. If the answer is not
present, say you could not find it in the provided documents.

Context:
{context}

Question: {question}
Answer:"""
)
rag_chain = prompt | llm


def ask_question(question):
    compressed_docs = retriever.invoke(question)
    if not compressed_docs:
        print("No relevant information found.")
        return

    context = "\n\n".join(doc.page_content for doc in compressed_docs)
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:\n" + response.content)

    print("\nRelevant sources:")
    for document in compressed_docs:
        print(f"- {document.metadata.get('source', 'Unknown source')}")


def main():
    print("Ask questions about your PDF documents. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question:
            try:
                ask_question(question)
            except Exception as error:
                print(f"Error: {error}")


if __name__ == "__main__":
    main()
