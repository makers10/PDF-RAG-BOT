# rag_pipeline.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import sys

# Windows-specific DLL loading fix for PyTorch/Transformers
if sys.platform == "win32":
    possible_venv_paths = [
        os.path.join(os.getcwd(), "venv", "Lib", "site-packages", "torch", "lib"),
        os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "torch", "lib")
    ]
    for lib_path in possible_venv_paths:
        if os.path.exists(lib_path):
            try:
                os.add_dll_directory(lib_path)
            except Exception:
                pass

try:
    import torch
except Exception as e:
    print(f"Early torch import failed in rag_pipeline: {e}")

import pickle
import re
import logging
from typing import List, Tuple, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from utils.pdf_loader import load_pdf
from utils.chunking import chunk_text
from utils.semantic_chunking import semantic_chunk_text, hybrid_chunk_text
from utils.embedding import get_embedding_model
from cache.redis_cache import get_cache
from retrieval.multi_level_retriever import create_multi_level_retriever
from raptor.raptor_tree import create_raptor_tree

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize cache (will use MockCache if Redis not available)
cache = get_cache(use_redis=True, ttl=86400)  # 24 hours TTL

logger.info(f"✅ Cache initialized: {cache.__class__.__name__}")
logger.info(f"📊 Cache stats: {cache.get_stats()}")

# -------------------- Text QA Model (OpenRouter with fallback) --------------------
# List of free models to try (covering different providers to avoid shared limits)
FREE_MODELS = [
    "openrouter/free", # Best automatic selection
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

import time as _time

def get_llm(model_name=None):
    """Create an LLM instance for the given model."""
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name=model_name or FREE_MODELS[0],
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "PDF RAG Bot"
        },
        temperature=0.7, # Higher temp can sometimes help get past simple cached 429s
        max_tokens=600,
        request_timeout=45,
        max_retries=0 # We handle retries manually
    )

# Default LLM
llm = get_llm()

# -------------------- Query Enhancement --------------------
def enhance_query(query: str) -> str:
    """
    Enhance user query for better retrieval:
    - Expand abbreviations
    - Add context keywords
    - Normalize formatting
    """
    query = query.strip()

    # Common abbreviations expansion
    abbreviations = {
        r'\bwhat\'s\b': 'what is',
        r'\bhow\'s\b': 'how is',
        r'\bwhere\'s\b': 'where is',
        r'\bwho\'s\b': 'who is',
        r'\bcan\'t\b': 'cannot',
        r'\bwon\'t\b': 'will not',
        r'\bdon\'t\b': 'do not',
    }

    for abbr, full in abbreviations.items():
        query = re.sub(abbr, full, query, flags=re.IGNORECASE)

    return query

# -------------------- Hybrid Search --------------------
def hybrid_search(vector_store, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
    """
    Hybrid search combining:
    1. Semantic search (vector similarity)
    2. Keyword matching (BM25-like)
    """
    # Semantic search with scores
    semantic_results = vector_store.similarity_search_with_score(query, k=top_k * 2)

    # Keyword boosting: if query words appear in chunk, boost its score
    query_words = set(query.lower().split())
    boosted_results = []

    for doc, score in semantic_results:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words.intersection(doc_words))

        # Boost score if keywords match (lower score = better in FAISS)
        keyword_boost = overlap * 0.05
        adjusted_score = score - keyword_boost

        boosted_results.append((doc, adjusted_score))

    # Sort by adjusted score and return top_k
    boosted_results.sort(key=lambda x: x[1])
    return boosted_results[:top_k]

# -------------------- Context Reranking --------------------
def rerank_contexts(query: str, docs_with_scores: List[Tuple[Document, float]]) -> List[Document]:
    """
    Rerank retrieved documents based on:
    - Similarity score
    - Query term frequency
    - Document position diversity
    """
    query_terms = query.lower().split()
    reranked = []

    for doc, score in docs_with_scores:
        content_lower = doc.page_content.lower()

        # Calculate term frequency score
        tf_score = sum(content_lower.count(term) for term in query_terms)

        # Combined score (lower is better for FAISS distance)
        combined_score = score - (tf_score * 0.02)

        reranked.append((doc, combined_score))

    # Sort by combined score
    reranked.sort(key=lambda x: x[1])

    return [doc for doc, _ in reranked]

# -------------------- Answer Verification --------------------
def verify_answer(answer: str, context: str, query: str) -> Tuple[str, float]:
    """
    Verify answer quality and add confidence score:
    - Check if answer is grounded in context
    - Detect hallucinations
    - Calculate confidence
    """
    answer_lower = answer.lower()
    context_lower = context.lower()

    # Check for common hallucination patterns
    hallucination_phrases = [
        "i don't know", "i cannot", "not mentioned",
        "doesn't provide", "no information", "unclear"
    ]

    is_uncertain = any(phrase in answer_lower for phrase in hallucination_phrases)

    # Calculate confidence based on answer-context overlap
    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())
    overlap = len(answer_words.intersection(context_words))

    if len(answer_words) > 0:
        confidence = min(overlap / len(answer_words), 1.0)
    else:
        confidence = 0.0

    # Reduce confidence if uncertain
    if is_uncertain:
        confidence *= 0.5

    # Add confidence indicator to answer
    if confidence > 0.7:
        confidence_label = "High confidence"
    elif confidence > 0.4:
        confidence_label = "Medium confidence"
    else:
        confidence_label = "Low confidence"

    return answer, confidence

# -------------------- Multi-hop Reasoning --------------------
def extract_key_facts(context: str) -> List[str]:
    """
    Extract key facts from context for multi-hop reasoning
    """
    # Split by sentences
    sentences = re.split(r'[.!?]+', context)

    # Filter meaningful sentences (length > 20 chars)
    facts = [s.strip() for s in sentences if len(s.strip()) > 20]

    return facts[:5]  # Top 5 facts

# -------------------- Vector Store Creation --------------------
def create_vectorstore_from_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    use_cache: bool = True,
    use_semantic_chunking: bool = False,
    use_multi_level: bool = False,
    use_raptor: bool = False,
    raptor_max_levels: int = 3
):
    """
    Create vector store with all Phase features:
    
    Phase 1:
    - BGE-Large embeddings (1024d)
    - Redis caching (24h TTL)
    
    Phase 2:
    - Semantic chunking (LlamaIndex)
    - Multi-level retrieval (BM25 + FAISS + Reranker)
    
    Phase 3:
    - RAPTOR tree (hierarchical retrieval)
    - Multi-level summarization
    """
    pdf_name = os.path.basename(pdf_path)
    vector_store_file = f"{pdf_name}.pkl"

    if use_cache and os.path.exists(vector_store_file):
        with open(vector_store_file, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                vector_store = data.get("vector_store")
                multi_level_retriever = data.get("multi_level_retriever")
                raptor_tree = data.get("raptor_tree")
            else:
                # Compatibility for old format (direct FAISS object)
                vector_store = data
                multi_level_retriever = None
                raptor_tree = None
            logger.info("✅ Loaded cached vector store")
            return vector_store, multi_level_retriever, raptor_tree
    logger.info(f"📂 Loading PDF: {pdf_name}")
    text = load_pdf(pdf_path)
    
    if not text or len(text.strip()) < 10:
        logger.error(f"❌ Failed to extract text from {pdf_name}. PDF might be empty or a scan.")
        return None, None, None

    # Phase 2A: Semantic chunking
    if use_semantic_chunking:
        logger.info("🔧 Using semantic chunking...")
        chunks = hybrid_chunk_text(text, use_semantic=True)
    else:
        logger.info("🔧 Using recursive chunking...")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=200)
    
    if not chunks:
        logger.error(f"❌ No chunks created for {pdf_name}.")
        return None, None, None

    logger.info(f"📦 Created {len(chunks)} chunks")

    # Add metadata to chunks
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": pdf_name}
        )
        documents.append(doc)

    if not documents:
        logger.error(f"❌ No documents created for {pdf_name}.")
        return None, None, None

    # Create vector store with BGE-Large embeddings
    try:
        logger.info("🔧 Creating embeddings with BGE-Large...")
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(documents, embedding_model)
    except Exception as e:
        logger.error(f"❌ FAISS creation failed: {e}")
        return None, None, None
    
    # Phase 2B: Multi-level retriever
    multi_level_retriever = None
    if use_multi_level:
        logger.info("🔧 Building multi-level retriever...")
        try:
            multi_level_retriever = create_multi_level_retriever(
                documents=[doc.page_content for doc in documents],
                vector_store=vector_store,
                use_reranker=True
            )
        except Exception as e:
            logger.warning(f"⚠️ Multi-level retriever building failed: {e}")
    
    # Phase 3: RAPTOR tree
    raptor_tree = None
    if use_raptor:
        logger.info("🔧 Building RAPTOR tree...")
        try:
            raptor_tree = create_raptor_tree(
                texts=chunks,
                embedding_model=embedding_model,
                max_levels=raptor_max_levels
            )
            
            # Log tree stats
            stats = raptor_tree.get_tree_stats()
            logger.info(f"📊 RAPTOR tree stats: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR tree building failed: {e}")

    # Cache
    try:
        with open(vector_store_file, "wb") as f:
            pickle.dump({
                "vector_store": vector_store,
                "multi_level_retriever": multi_level_retriever,
                "raptor_tree": raptor_tree
            }, f)
        logger.info("✅ Vector store created and cached with RAPTOR tree")
    except Exception as e:
        logger.error(f"❌ Failed to cache vector store: {e}")

    return vector_store, multi_level_retriever, raptor_tree

# -------------------- Advanced Answer Generation --------------------
def answer_question(
    vector_store, 
    query: str, 
    top_k: int = 5,
    similarity_threshold: float = 1.5,
    use_hybrid: bool = True,
    return_sources: bool = False,
    return_context: bool = False,
    use_cache: bool = True,
    multi_level_retriever = None,
    use_multi_level: bool = True,
    raptor_tree = None,
    use_raptor: bool = True,
    raptor_collapse_tree: bool = True
) -> str:
    """
    Advanced RAG with all 3 Phases:
    
    Phase 1:
    - BGE-Large embeddings (1024d)
    - Redis caching (24h TTL)
    - Query enhancement
    - Answer verification
    
    Phase 2:
    - Semantic chunking (LlamaIndex)
    - Multi-level retrieval (BM25 + FAISS + Reranker)
    - Ensemble scoring
    
    Phase 3:
    - RAPTOR tree (hierarchical retrieval)
    - Multi-level summarization
    - Tree traversal
    """

    # Step 0: Check cache first (Phase 1)
    if use_cache:
        cached_result = cache.get_answer(query)
        if cached_result:
            logger.info(f"🎯 Cache HIT for query: {query[:50]}...")
            answer = cached_result.get("answer", "")
            if answer:
                if return_context:
                    return {"answer": answer, "context": "", "sources": []}
                return answer
        else:
            logger.info(f"❌ Cache MISS for query: {query[:50]}...")

    # Step 1: Enhance query
    enhanced_query = enhance_query(query)

    # Step 2: Retrieval Strategy Selection
    # Priority: RAPTOR > Multi-level > Hybrid > Semantic
    
    if use_raptor and raptor_tree:
        logger.info("🌳 Using RAPTOR tree retrieval...")
        tree_results = raptor_tree.retrieve_from_tree(
            enhanced_query,
            top_k=top_k,
            collapse_tree=raptor_collapse_tree
        )
        # Convert to (doc, score) format
        results = [(Document(page_content=text, metadata={"level": level}), 1.0 - score) for text, score, level in tree_results]
        
    elif use_multi_level and multi_level_retriever:
        logger.info("🔍 Using multi-level retrieval...")
        ml_results = multi_level_retriever.retrieve_multi_level(
            enhanced_query,
            top_k=top_k,
            bm25_weight=0.3,
            semantic_weight=0.7
        )
        # Convert to (doc, score) format
        results = [(Document(page_content=doc), score) for doc, score in ml_results]
        
    elif use_hybrid:
        logger.info("🔍 Using hybrid search...")
        results = hybrid_search(vector_store, enhanced_query, top_k=top_k)
        
    else:
        logger.info("🔍 Using semantic search...")
        results = vector_store.similarity_search_with_score(enhanced_query, k=top_k)

    # Step 3: Filter by similarity threshold
    relevant_docs = [(doc, score) for doc, score in results if score < similarity_threshold]

    if not relevant_docs:
        # If no docs meet threshold, use top results anyway
        relevant_docs = results[:min(2, len(results))]  # Use at least top 2 results

    # Step 4: Rerank contexts
    reranked_docs = rerank_contexts(enhanced_query, relevant_docs)

    # Step 5: Build structured context
    context_parts = []
    source_info = []

    for i, doc in enumerate(reranked_docs, 1):
        context_parts.append(f"[Passage {i}]: {doc.page_content}")
        if hasattr(doc, 'metadata'):
            source_info.append(f"Passage {i}: Chunk {doc.metadata.get('chunk_id', 'N/A')}")

    context = "\n\n".join(context_parts)

    # Step 6: Context is already built in Step 5. No manual T5-style truncation needed.

    # Step 7: Extract key facts for reasoning
    key_facts = extract_key_facts(context)

    # Step 8: Enhanced prompt (User-only role for maximum compatibility)
    prompt_template = ChatPromptTemplate.from_messages([
        ("user", "System: You are a precise and helpful assistant. Use the provided context and key facts to answer the question. Only use the provided information.\n\nContext:\n{context}\n\nKey Facts:\n{facts}\n\nQuestion: {query}")
    ])

    # Step 9: Generate answer using OpenRouter (with model fallback)
    facts_str = "\n".join([f"- {fact}" for fact in key_facts])
    answer = None
    
    for model_name in FREE_MODELS:
        try:
            logger.info(f"🤖 Requesting answer from {model_name}...")
            # Create a fresh LLM instance to ensure fresh configuration/timeout
            api_llm = get_llm(model_name)
            chain = prompt_template | api_llm
            response = chain.invoke({
                "context": context,
                "facts": facts_str,
                "query": query
            })
            answer = response.content.strip()
            logger.info(f"✅ Success with {model_name}!")
            break
        except Exception as e:
            logger.warning(f"⚠️ {model_name} rate-limited or error: {type(e).__name__}")
            _time.sleep(7)  # Seven seconds to reset limits
            continue
    
    if not answer:
        logger.error("❌ All LLM models failed. Using context fallback.")
        if context:
            answer = f"Based on the document: {context[:500]}"
        else:
            answer = "I couldn't generate an answer. Please try again later."

    # Step 11: Verify answer quality (Optional, as Llama 3 is much better)
    answer, confidence = verify_answer(answer, context, query)

    # Step 12: Format final response
    final_answer = answer if answer and len(answer) > 5 else "I couldn't generate a proper answer from the available context."

    # Step 13: Cache the result
    if use_cache and final_answer:
        cache.cache_answer(query, final_answer, metadata={
            "sources": source_info if return_sources else [],
            "confidence": confidence
        })
        logger.info(f"💾 Cached answer for query: {query[:50]}...")

    if return_context:
        return {"answer": final_answer, "context": context, "sources": source_info}

    return final_answer

# -------------------- Optional: PDF to Images --------------------
def pdf_to_images(pdf_path: str):
    """
    Convert PDF pages to images (optional, for visual RAG).
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=200)
        return images
    except ImportError:
        print("pdf2image not installed. Install via `pip install pdf2image`")
        return []
