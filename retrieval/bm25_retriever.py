# retrieval/bm25_retriever.py
"""
BM25 Keyword-based Retrieval
- Best for exact term matching
- Complements semantic search
- Fast and efficient
"""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Matching 25) keyword-based retrieval
    
    How it works:
    - Tokenizes documents
    - Builds inverted index
    - Scores based on term frequency and document frequency
    - Great for exact keyword matches
    """
    
    def __init__(self, documents: List[str]):
        """
        Initialize BM25 index
        
        Args:
            documents: List of text documents
        """
        self.documents = documents
        
        # Tokenize documents (simple whitespace tokenization)
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        logger.info(f"✅ BM25 index built with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return documents with scores
        results = [
            (self.documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # Only return documents with non-zero scores
        ]
        
        logger.debug(f"BM25 retrieved {len(results)} documents")
        
        return results
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for all documents
        
        Args:
            query: Search query
            
        Returns:
            Array of scores
        """
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)


def create_bm25_index(documents: List[str]) -> BM25Retriever:
    """
    Factory function to create BM25 retriever
    
    Args:
        documents: List of text documents
        
    Returns:
        BM25Retriever instance
    """
    return BM25Retriever(documents)
