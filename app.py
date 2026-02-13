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
            st.info("Searching for answer...")

            # Display Lottie animation while waiting
            if lottie_search:
                st_lottie(lottie_search, height=200, key="searching")

            # Run advanced RAG with all optimizations
            answer = answer_question(
                vector_store, 
                query, 
                top_k=4, 
                similarity_threshold=0.3,
                use_hybrid=True,
                return_sources=True
            )
            st.markdown("**Answer:**")
            st.write(answer)


# # app.py old one
# import streamlit as st
# import os
# import pickle
# import time
# import requests
# from streamlit_lottie import st_lottie
# from rag_pipeline import create_vectorstore_from_pdf, answer_question

# # -------------------- Streamlit Page Setup --------------------
# st.set_page_config(
#     page_title="PDF RAG Bot",
#     page_icon="📄",
#     layout="wide"
# )
# st.title("📄 PDF RAG Bot - Extractive Answers")
# st.write("Upload a PDF, and ask questions. Answers are extractive and concise!")

# # -------------------- Lottie Animation Loader --------------------
# def load_lottie_url(url: str):
#     """Load Lottie animation from a URL."""
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # URL of a searching/processing Lottie animation
# lottie_searching = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")  # magnifying glass

# # -------------------- PDF Upload --------------------
# uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded_file is not None:
#     pdf_name = uploaded_file.name
#     vector_store_file = f"{pdf_name}.pkl"

#     # -------------------- Spinner & Status --------------------
#     status_text = st.empty()
#     progress_bar = st.progress(0)

#     # -------------------- Load or Create Vector Store --------------------
#     if os.path.exists(vector_store_file):
#         status_text.text("Loading cached vector store...")
#         with open(vector_store_file, "rb") as f:
#             vector_store = pickle.load(f)
#         st.success("Vector store loaded from cache ✅")
#     else:
#         status_text.text("Creating vector store from PDF...")
#         # Save uploaded PDF temporarily
#         with open(pdf_name, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Function to create vector store with progress
#         def vectorstore_with_progress(file_path):
#             from utils.pdf_loader import load_pdf
#             from utils.chunking import chunk_text
#             from utils.embedding import get_embedding_model
#             from langchain_community.vectorstores.faiss import FAISS

#             text = load_pdf(file_path)
#             status_text.text("Chunking PDF text...")
#             chunks = chunk_text(text, chunk_size=300)
#             status_text.text("Embedding chunks...")
#             embedding_model = get_embedding_model()

#             vector_store_chunks = []
#             for i, chunk in enumerate(chunks):
#                 vector_store_chunks.append(chunk)
#                 # update progress bar
#                 progress_bar.progress((i + 1) / len(chunks))
#                 time.sleep(0.02)  # optional, slow for UI effect

#             vector_store = FAISS.from_texts(vector_store_chunks, embedding_model)
#             return vector_store

#         vector_store = vectorstore_with_progress(pdf_name)

#         # Save vector store to pickle for future use
#         with open(vector_store_file, "wb") as f:
#             pickle.dump(vector_store, f)
#         st.success("Vector store created and cached ✅")

#     # -------------------- Question Input --------------------
#     st.markdown("---")
#     query = st.text_input("Ask a question about this PDF:")

#     if query:
#         # Placeholder for Lottie animation while searching
#         lottie_placeholder = st.empty()
#         with st.spinner("Finding answer... 🔍"):
#             # Display Lottie animation
#             lottie_placeholder.lottie(
#                 lottie_searching,
#                 speed=1,
#                 width=200,
#                 height=200,
#                 key="searching"
#             )
#             # simulate small delay for visual effect
#             time.sleep(0.5)

#             # Retrieve answer
#             answer = answer_question(vector_store, query)

#         # Clear the animation
#         lottie_placeholder.empty()

#         # Display the answer
#         st.markdown("**Answer:**")
#         st.write(answer)
