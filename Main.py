from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load Chroma Vector Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)

# Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply exactly:

"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("=" * 50)
print("RAG System Ready!")
print("Type 0 to exit.")
print("=" * 50)

while True:

    query = input("\nYou: ").strip()

    if query == "0":
        print("Goodbye!")
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    if len(docs) == 0:
        print("\nAI: I could not find the answer in the document.")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)
