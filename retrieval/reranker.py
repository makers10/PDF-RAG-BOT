# retrieval/reranker.py
"""
Cross-Encoder Reranking
- Reranks retrieved documents
- More accurate than bi-encoder
- Final quality layer
"""

import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for final ranking
    
    How it works:
    - Takes query + document pairs
    - Scores each pair with cross-attention
    - More accurate than bi-encoder (but slower)
    - Use as final reranking step
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder
        
        Args:
            model_name: HuggingFace model name
        """
        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"✅ Reranker loaded: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load reranker: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples, sorted by score
        """
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            results = doc_scores[:top_k]
            
            logger.debug(f"Reranked {len(documents)} → {len(results)} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]
    
    def rerank_with_scores(
        self,
        query: str,
        doc_score_pairs: List[Tuple[str, float]],
        top_k: int = 5,
        combine_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents that already have scores
        
        Args:
            query: Search query
            doc_score_pairs: List of (document, original_score) tuples
            top_k: Number of results to return
            combine_scores: Combine original and reranker scores
            
        Returns:
            List of (document, combined_score) tuples
        """
        if not self.model or not doc_score_pairs:
            return doc_score_pairs[:top_k]
        
        try:
            documents = [doc for doc, _ in doc_score_pairs]
            original_scores = [score for _, score in doc_score_pairs]
            
            # Get reranker scores
            pairs = [[query, doc] for doc in documents]
            reranker_scores = self.model.predict(pairs)
            
            # Combine scores
            if combine_scores:
                # Normalize scores to 0-1 range
                orig_min, orig_max = min(original_scores), max(original_scores)
                rerank_min, rerank_max = min(reranker_scores), max(reranker_scores)
                
                if orig_max > orig_min:
                    norm_orig = [(s - orig_min) / (orig_max - orig_min) for s in original_scores]
                else:
                    norm_orig = [0.5] * len(original_scores)
                
                if rerank_max > rerank_min:
                    norm_rerank = [(s - rerank_min) / (rerank_max - rerank_min) for s in reranker_scores]
                else:
                    norm_rerank = [0.5] * len(reranker_scores)
                
                # Weighted combination (70% reranker, 30% original)
                combined_scores = [
                    0.7 * r + 0.3 * o
                    for r, o in zip(norm_rerank, norm_orig)
                ]
            else:
                combined_scores = reranker_scores
            
            # Combine with documents
            results = list(zip(documents, combined_scores))
            
            # Sort by combined score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking with scores failed: {e}")
            return doc_score_pairs[:top_k]


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """
    Factory function to create reranker
    
    Args:
        model_name: HuggingFace cross-encoder model
        
    Returns:
        CrossEncoderReranker instance
    """
    return CrossEncoderReranker(model_name)
