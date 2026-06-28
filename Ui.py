import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📖",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #4f8ef7;
    --accent-glow: rgba(79,142,247,0.18);
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --error:     #f85149;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* ── Main container ── */
[data-testid="stMain"] > div {
    padding-top: 2.5rem !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon {
    font-size: 2.6rem;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.01em;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 0 0 0 var(--accent-glow);
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 0 18px var(--accent-glow) !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.15s ease;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
}

/* ── Alert / success banners ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.875rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    color: var(--accent) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    box-shadow: 0 0 24px var(--accent-glow);
    position: relative;
}
.answer-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
}
.answer-text {
    color: var(--text);
    font-size: 0.97rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ── Status dot ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--success);
    font-weight: 500;
    margin-bottom: 1.2rem;
}
.dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Chunk count badge ── */
.badge {
    display: inline-block;
    background: rgba(79,142,247,0.12);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.18rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">📖</div>
    <h1>Book Assistant</h1>
    <p>Upload a PDF document and ask questions — answers are grounded in the text.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 — Upload Document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a PDF here or click to browse",
    type="pdf",
    label_visibility="collapsed",
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"**{uploaded_file.name}** uploaded successfully.")

    st.markdown('<div class="section-label" style="margin-top:1.4rem;">Step 2 — Index Document</div>', unsafe_allow_html=True)
    if st.button("⚙  Build Vector Index", use_container_width=False):
        with st.spinner("Chunking and embedding document…"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db",
            )
            vectorstore.persist()
        st.success(f"Index ready — **{len(chunks)} chunks** stored.")

# ── Q&A section ───────────────────────────────────────────────────────────────
if os.path.exists("chroma_db"):
    st.markdown("<hr>", unsafe_allow_html=True)

    # Status indicator
    st.markdown("""
    <div class="status-row">
        <div class="dot"></div>
        Vector index loaded and ready
    </div>
    """, unsafe_allow_html=True)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a precise document assistant.
Answer using ONLY the provided context.
If the answer is not present, respond: "I could not find the answer in the document."
Be concise and factual.""",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ])

    st.markdown('<div class="section-label">Step 3 — Ask a Question</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Question",
        placeholder="What is this document about?",
        label_visibility="collapsed",
    )

    if query:
        with st.spinner("Retrieving context and generating answer…"):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">AI Answer</div>
            <div class="answer-text">{response.content}</div>
        </div>
        """, unsafe_allow_html=True)
