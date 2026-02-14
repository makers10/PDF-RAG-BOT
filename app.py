# app.py
import streamlit as st
import os
import pickle
import requests
from streamlit_lottie import st_lottie
from rag_pipeline import create_vectorstore_from_pdf, answer_question

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(
    page_title="PDF RAG Bot",
    page_icon="📄",
    layout="wide"
)
st.title("📄 PDF RAG Bot - Extractive Answers")
st.write("Upload a PDF and ask questions. Answers are extractive and concise!")

# -------------------- Helper: Load Lottie Animation --------------------
def load_lottie_url(url: str) -> dict:
    """Fetch Lottie JSON from URL and ensure it returns a dict."""
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print("Failed to load Lottie animation:", e)
    return {}

# Lottie animation URL (searching PDF)
LOTTIE_SEARCH_URL = "https://assets9.lottiefiles.com/packages/lf20_xyz.json"
lottie_search = load_lottie_url(LOTTIE_SEARCH_URL)

# -------------------- PDF Upload --------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_name = uploaded_file.name
    vector_store_file = f"{pdf_name}.pkl"

    # -------------------- Spinner & Progress --------------------
    status_text = st.empty()
    progress_bar = st.progress(0)

    # -------------------- Load or Create Vector Store --------------------
    if os.path.exists(vector_store_file):
        status_text.text("Loading cached vector store...")
        with open(vector_store_file, "rb") as f:
            vector_store = pickle.load(f)
        st.success("Vector store loaded from cache ✅")
    else:
        status_text.text("Creating vector store from PDF...")
        # Save uploaded PDF to local disk
        with open(pdf_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create vector store with advanced chunking
        vector_store = create_vectorstore_from_pdf(pdf_name, chunk_size=800)

        # Save vector store for future use
        with open(vector_store_file, "wb") as f:
            pickle.dump(vector_store, f)
        st.success("Vector store created and cached ✅")

    # -------------------- Question Input --------------------
    st.markdown("---")
    query = st.text_input("Ask a question about this PDF:")

    if query:
        with st.container():
            st.info("🔍 Searching for answer...")

            # Display Lottie animation while waiting
            if lottie_search:
                st_lottie(lottie_search, height=200, key="searching")

            # Run advanced RAG with all optimizations
            result = answer_question(
                vector_store,
                query,
                top_k=5,
                similarity_threshold=1.5,
                use_hybrid=True,
                return_sources=False,
                return_context=False
            )

            # Display answer in a nice format
            st.markdown("---")
            st.markdown("### 💡 Answer")
            st.write(result)
