
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

CHROMA_DB_DIR = "./chroma_db"

# 1. Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 2. Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder
)

from langchain_classic.retrievers import (
    ContextualCompressionRetriever
)

from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker
)


# # 1. Initialize cross-encoder
# model = HuggingFaceCrossEncoder(
#     model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}
# )

# # 2. Create reranker
# compressor = CrossEncoderReranker(
#     model=model,
#     top_n=3
# )

# # 3. Initial vector retrieval
# base_retriever = vector_store.as_retriever(
#     search_kwargs={"k": 10}
# )

# # 4. Combine retriever and reranker
# compression_retriever = ContextualCompressionRetriever(
#     base_retriever=base_retriever,
#     base_compressor=compressor
# )

# # 5. Retrieve reranked documents
# question = "What are the security measures?"

# documents = compression_retriever.invoke(question)

# for doc in documents:
#     print(doc.page_content)

# 3. Create retriever
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "fetch_k": 10,
#         "lambda_mult": 0.5
#     }
# )

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,
        "k": 3
    }
)

# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 3,
#         "filter": {
#             "$and": [
#                 {"category": "HR"},
#                 {"source": "leave_policy.pdf"}
#             ]
#         }
#     }
# )
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 5,
#         "filter": {
#             "$and": [
#                 {"source": "leave_policy.pdf"},
#                 {"page": {"$gte": 9}},
#                 {"page": {"$lte": 19}}
#             ]
#         }
#     }
# )

# 4. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# 5. Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer the question using ONLY the
    information provided in the context.

    If the answer is not present in the context,
    say that you could not find the answer
    in the provided documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# 6. Create RAG chain
rag_chain = prompt | llm


# 7. Ask questions

# Dynamic retrieval with metadata filtering
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

def rerank_documents(question, documents):

    # Create query-document pairs
    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    # Generate relevance scores
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    ranked_documents = list(zip(documents, scores))

    # Sort by score: highest first
    ranked_documents.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    # Return reranked documents
    return ranked_documents


def ask_question(question):

    # STEP 1: Retrieve top 10 candidates
    retrieved_docs = vector_store.similarity_search(
        query=question,
        k=10
    )

    if not retrieved_docs:
        print("No documents found.")
        return

    print(f"\nRetrieved {len(retrieved_docs)} chunks")

    # STEP 2: Rerank candidates
    ranked_documents = rerank_documents(
        question,
        retrieved_docs
    )

    # STEP 3: Select top 3
    top_documents = ranked_documents[:3]

    print("\nReranked results:")
    print("-" * 50)

    for rank, (doc, score) in enumerate(
        top_documents, start=1
    ):

        print(f"\nRank: {rank}")
        print(f"Score: {float(score):.4f}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:200]}")

    # STEP 4: Prepare context
    context = "\n\n".join(
        doc.page_content
        for doc, score in top_documents
    )

    # STEP 5: Generate answer using Gemini
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nFinal Answer:")
    print("-" * 50)
    print(response.content)



# 8. Terminal application
def main():

    print("\n===== LANGCHAIN RAG APPLICATION =====")
    print("Ask questions about your PDF documents.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            ask_question(question)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
