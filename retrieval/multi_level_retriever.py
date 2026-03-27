# retrieval/multi_level_retriever.py
"""
Multi-Level Retrieval System
- Level 1: BM25 (keyword matching)
- Level 2: FAISS (semantic search)
- Level 3: Cross-encoder reranking
- Ensemble scoring for best results
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class MultiLevelRetriever:
    """
    Multi-level retrieval combining multiple strategies
    
    Architecture:
    1. BM25 (keyword) - Fast, exact matches
    2. FAISS (semantic) - Meaning-based
    3. Reranker (cross-encoder) - Final quality check
    4. Ensemble - Combine all scores
    """
    
    def __init__(
        self,
        documents: List[str],
        vector_store,
        use_reranker: bool = True
    ):
        """
        Initialize multi-level retriever
        
        Args:
            documents: List of text documents
            vector_store: FAISS vector store
            use_reranker: Enable cross-encoder reranking
        """
        self.documents = documents
        self.vector_store = vector_store
        
        # Level 1: BM25
        logger.info("🔧 Building BM25 index...")
        self.bm25 = BM25Retriever(documents)
        
        # Level 3: Reranker
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info("🔧 Loading reranker...")
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        logger.info("✅ Multi-level retriever initialized")
    
    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Level 1: BM25 keyword search"""
        return self.bm25.retrieve(query, top_k=top_k)
    
    def retrieve_semantic(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Level 2: Semantic search with FAISS"""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, float(score)) for doc, score in results]
    
    def ensemble_scores(
        self,
        bm25_results: List[Tuple[str, float]],
        semantic_results: List[Tuple[str, float]],
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Combine BM25 and semantic scores
        
        Args:
            bm25_results: BM25 results
            semantic_results: Semantic results
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
            
        Returns:
            Combined results with ensemble scores
        """
        # Create score dictionaries
        bm25_scores = {doc: score for doc, score in bm25_results}
        semantic_scores = {doc: score for doc, score in semantic_results}
        
        # Get all unique documents
        all_docs = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Normalize scores
        bm25_vals = list(bm25_scores.values())
        semantic_vals = list(semantic_scores.values())
        
        bm25_max = max(bm25_vals) if bm25_vals else 1.0
        semantic_max = max(semantic_vals) if semantic_vals else 1.0
        
        # Combine scores
        ensemble_results = []
        for doc in all_docs:
            bm25_score = bm25_scores.get(doc, 0.0) / bm25_max if bm25_max > 0 else 0.0
            
            # For FAISS, lower score is better (L2 distance)
            # Invert and normalize
            semantic_score = semantic_scores.get(doc, semantic_max)
            semantic_score = 1.0 - (semantic_score / semantic_max) if semantic_max > 0 else 0.0
            
            # Weighted combination
            combined_score = (bm25_weight * bm25_score) + (semantic_weight * semantic_score)
            
            ensemble_results.append((doc, combined_score))
        
        # Sort by combined score (descending)
        ensemble_results.sort(key=lambda x: x[1], reverse=True)
        
        return ensemble_results
    
    def retrieve_multi_level(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        intermediate_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Multi-level retrieval with all strategies
        
        Args:
            query: Search query
            top_k: Final number of results
            bm25_weight: Weight for BM25 (keyword)
            semantic_weight: Weight for semantic
            intermediate_k: Number of candidates from each method
            
        Returns:
            Top-k documents with scores
        """
        logger.info(f"🔍 Multi-level retrieval for: {query[:50]}...")
        
        # Level 1: BM25 (keyword)
        logger.debug("Level 1: BM25 search...")
        bm25_results = self.retrieve_bm25(query, top_k=intermediate_k)
        logger.debug(f"  → {len(bm25_results)} BM25 results")
        
        # Level 2: Semantic (FAISS)
        logger.debug("Level 2: Semantic search...")
        semantic_results = self.retrieve_semantic(query, top_k=intermediate_k)
        logger.debug(f"  → {len(semantic_results)} semantic results")
        
        # Ensemble scoring
        logger.debug("Combining scores...")
        ensemble_results = self.ensemble_scores(
            bm25_results,
            semantic_results,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight
        )
        
        # Get top candidates for reranking
        top_candidates = ensemble_results[:top_k * 2]
        
        # Level 3: Reranking (optional)
        if self.use_reranker and self.reranker:
            logger.debug("Level 3: Reranking...")
            final_results = self.reranker.rerank_with_scores(
                query,
                top_candidates,
                top_k=top_k,
                combine_scores=True
            )
            logger.debug(f"  → {len(final_results)} reranked results")
        else:
            final_results = top_candidates[:top_k]
        
        logger.info(f"✅ Retrieved {len(final_results)} final results")
        
        return final_results
    
    def get_retrieval_stats(self, query: str) -> Dict:
        """
        Get statistics for each retrieval method
        
        Returns:
            Dict with stats for each level
        """
        bm25_results = self.retrieve_bm25(query, top_k=10)
        semantic_results = self.retrieve_semantic(query, top_k=10)
        
        return {
            "bm25": {
                "count": len(bm25_results),
                "avg_score": np.mean([s for _, s in bm25_results]) if bm25_results else 0,
                "max_score": max([s for _, s in bm25_results]) if bm25_results else 0
            },
            "semantic": {
                "count": len(semantic_results),
                "avg_score": np.mean([s for _, s in semantic_results]) if semantic_results else 0,
                "min_score": min([s for _, s in semantic_results]) if semantic_results else 0
            }
        }


def create_multi_level_retriever(
    documents: List[str],
    vector_store,
    use_reranker: bool = True
) -> MultiLevelRetriever:
    """
    Factory function to create multi-level retriever
    
    Args:
        documents: List of text documents
        vector_store: FAISS vector store
        use_reranker: Enable reranking
        
    Returns:
        MultiLevelRetriever instance
    """
    return MultiLevelRetriever(documents, vector_store, use_reranker)
