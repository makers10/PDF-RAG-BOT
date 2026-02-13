# rag_pipeline.py
import os
import pickle
import re
from typing import List, Tuple, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from utils.pdf_loader import load_pdf
from utils.chunking import chunk_text
from utils.embedding import get_embedding_model

# -------------------- Text QA Model --------------------
GEN_MODEL_NAME = "google/flan-t5-base"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

gen_pipeline = pipeline(
    task="text2text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer,
    device=-1
)

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
def create_vectorstore_from_pdf(pdf_path: str, chunk_size: int = 800, use_cache: bool = True):
    """
    Create vector store with metadata for better retrieval
    """
    pdf_name = os.path.basename(pdf_path)
    vector_store_file = f"{pdf_name}.pkl"

    if use_cache and os.path.exists(vector_store_file):
        with open(vector_store_file, "rb") as f:
            vector_store = pickle.load(f)
        return vector_store

    # Load and chunk text
    text = load_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=200)

    # Add metadata to chunks for better tracking
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": pdf_name}
        )
        documents.append(doc)

    # Create vector store with metadata
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Cache
    with open(vector_store_file, "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

# -------------------- Advanced Answer Generation --------------------
def answer_question(
    vector_store, 
    query: str, 
    top_k: int = 4, 
    similarity_threshold: float = 0.3,
    use_hybrid: bool = True,
    return_sources: bool = False
) -> str:
    """
    Advanced RAG with all optimizations:
    - Query enhancement
    - Hybrid search
    - Context reranking
    - Token-based truncation
    - Answer verification
    - Source citation
    """
    
    # Step 1: Enhance query
    enhanced_query = enhance_query(query)
    
    # Step 2: Hybrid search with scores
    if use_hybrid:
        results = hybrid_search(vector_store, enhanced_query, top_k=top_k)
    else:
        results = vector_store.similarity_search_with_score(enhanced_query, k=top_k)
    
    # Step 3: Filter by similarity threshold
    relevant_docs = [(doc, score) for doc, score in results if score < similarity_threshold]
    
    if not relevant_docs:
        return "I couldn't find sufficiently relevant information in the document to answer this question confidently."
    
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
    
    # Step 6: Token-based truncation
    context_tokens = gen_tokenizer.encode(context, add_special_tokens=False)
    max_context_tokens = 400
    
    if len(context_tokens) > max_context_tokens:
        context_tokens = context_tokens[:max_context_tokens]
        context = gen_tokenizer.decode(context_tokens, skip_special_tokens=True)
        context += "..."
    
    # Step 7: Extract key facts for reasoning
    key_facts = extract_key_facts(context)
    
    # Step 8: Enhanced prompt with reasoning
    prompt = f"""You are a precise question-answering assistant. Answer based strictly on the provided context.

Context from document:
{context}

Key Facts:
{chr(10).join(f"- {fact}" for fact in key_facts[:3])}

Question: {query}

Instructions:
- Answer using ONLY information from the context
- Be accurate, concise, and direct
- If context lacks information, state: "The document doesn't provide enough information to answer this question"
- Do not add external knowledge

Answer:"""

    # Step 9: Generate answer with optimal parameters
    result = gen_pipeline(
        prompt,
        max_new_tokens=100,
        min_new_tokens=5,
        do_sample=False,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.0
    )

    # Step 10: Extract and clean answer
    answer = result[0]["generated_text"].strip()
    
    # Remove common prefixes
    prefixes = ["Answer:", "A:", "The answer is:", "Based on the context,", "According to the document,"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Step 11: Verify answer quality
    answer, confidence = verify_answer(answer, context, query)
    
    # Step 12: Format final response
    if return_sources and source_info:
        answer += f"\n\n📚 Sources: {', '.join(source_info[:2])}"
    
    return answer if answer and len(answer) > 5 else "I couldn't generate a proper answer from the available context."

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
