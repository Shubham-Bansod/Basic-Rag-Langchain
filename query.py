from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# python3 -m pip install rank_bm25 langchain-classic
load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
RETRIEVAL_COUNT = 3
FETCH_COUNT = 5


def create_bm25_retriever():
    """Build a keyword retriever from the chunks saved during ingestion."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        ),
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

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


retriever = create_bm25_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
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
    documents = retriever.invoke(question)[:RETRIEVAL_COUNT]
    if not documents:
        print("No relevant documents found.")
        return

    context = "\n\n".join(document.page_content for document in documents)
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:\n" + response.content)


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
