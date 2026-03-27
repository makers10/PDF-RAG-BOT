# utils/semantic_chunking.py
"""
LlamaIndex Semantic Chunking
- Splits text at semantic boundaries
- Preserves sentence coherence
- Better than character-based splitting
"""

import logging
from typing import List
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document as LlamaDocument

logger = logging.getLogger(__name__)


def semantic_chunk_text(
    text: str,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[str]:
    """
    Semantic chunking using LlamaIndex
    
    Args:
        text: Input text to chunk
        buffer_size: Number of sentences to group together (1-3 recommended)
        breakpoint_percentile_threshold: Threshold for semantic breaks (90-99)
        embed_model_name: Embedding model for semantic similarity
        
    Returns:
        List of semantically coherent chunks
        
    How it works:
    1. Splits text into sentences
    2. Embeds each sentence
    3. Calculates similarity between consecutive sentences
    4. Breaks at low similarity points (semantic boundaries)
    """
    
    try:
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device="cpu"
        )
        
        # Create semantic splitter
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model
        )
        
        # Create LlamaIndex document
        document = LlamaDocument(text=text)
        
        # Split into nodes
        nodes = splitter.get_nodes_from_documents([document])
        
        # Extract text from nodes
        chunks = [node.text for node in nodes]
        
        logger.info(f"✅ Semantic chunking: {len(chunks)} chunks created")
        logger.info(f"📊 Avg chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Semantic chunking failed: {e}")
        logger.warning("⚠️ Falling back to recursive chunking")
        
        # Fallback to recursive chunking
        from utils.chunking import chunk_text
        return chunk_text(text, chunk_size=800, chunk_overlap=200)


def hybrid_chunk_text(
    text: str,
    use_semantic: bool = True,
    min_chunk_size: int = 20,
    max_chunk_size: int = 1500
) -> List[str]:
    """
    Hybrid chunking: Semantic + Size constraints
    
    Args:
        text: Input text
        use_semantic: Use semantic chunking (True) or recursive (False)
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        
    Returns:
        List of chunks with size constraints
    """
    
    # Get initial chunks
    if use_semantic:
        chunks = semantic_chunk_text(text)
    else:
        from utils.chunking import chunk_text
        chunks = chunk_text(text, chunk_size=800, chunk_overlap=200)
    
    # Apply size constraints
    final_chunks = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        
        # Skip very small chunks
        if len(chunk) < min_chunk_size:
            continue
        
        # Split very large chunks
        if len(chunk) > max_chunk_size:
            # Split at sentence boundaries
            sentences = chunk.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                final_chunks.append(current_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    logger.info(f"✅ Hybrid chunking: {len(final_chunks)} final chunks")
    
    return final_chunks


def compare_chunking_methods(text: str) -> dict:
    """
    Compare different chunking methods
    
    Returns:
        Dict with statistics for each method
    """
    from utils.chunking import chunk_text
    
    # Method 1: Recursive
    recursive_chunks = chunk_text(text, chunk_size=800, chunk_overlap=200)
    
    # Method 2: Semantic
    semantic_chunks = semantic_chunk_text(text)
    
    # Method 3: Hybrid
    hybrid_chunks = hybrid_chunk_text(text)
    
    return {
        "recursive": {
            "count": len(recursive_chunks),
            "avg_size": sum(len(c) for c in recursive_chunks) / len(recursive_chunks),
            "min_size": min(len(c) for c in recursive_chunks),
            "max_size": max(len(c) for c in recursive_chunks)
        },
        "semantic": {
            "count": len(semantic_chunks),
            "avg_size": sum(len(c) for c in semantic_chunks) / len(semantic_chunks),
            "min_size": min(len(c) for c in semantic_chunks),
            "max_size": max(len(c) for c in semantic_chunks)
        },
        "hybrid": {
            "count": len(hybrid_chunks),
            "avg_size": sum(len(c) for c in hybrid_chunks) / len(hybrid_chunks),
            "min_size": min(len(c) for c in hybrid_chunks),
            "max_size": max(len(c) for c in hybrid_chunks)
        }
    }
