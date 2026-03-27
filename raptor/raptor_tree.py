# raptor/raptor_tree.py
"""
RAPTOR Tree Implementation
- Hierarchical document structure
- Multi-level retrieval
- Abstractive summarization at each level
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from raptor.clustering import RAPTORClusterer
from raptor.summarizer import RAPTORSummarizer

logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    """
    Node in RAPTOR tree
    
    Attributes:
        text: Node content (original chunk or summary)
        embedding: Vector embedding
        level: Tree level (0=leaf, 1+=internal)
        children: Child node indices
        cluster_id: Cluster identifier
        is_summary: Whether this is a summary node
    """
    text: str
    embedding: np.ndarray
    level: int
    children: List[int] = None
    cluster_id: int = 0
    is_summary: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class RAPTORTree:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    
    Architecture:
    Level 0 (Leaves): Original document chunks
    Level 1: Summaries of clusters of chunks
    Level 2: Summaries of Level 1 summaries
    Level 3: Document-level summary (root)
    
    Retrieval:
    - Can search at any level
    - Can traverse tree top-down or bottom-up
    - Combines results from multiple levels
    """
    
    def __init__(
        self,
        embedding_model,
        max_levels: int = 3,
        reduction_dimension: int = 10,
        min_cluster_size: int = 3
    ):
        """
        Initialize RAPTOR tree
        
        Args:
            embedding_model: Embedding model for vectors
            max_levels: Maximum tree depth
            reduction_dimension: UMAP dimensions for clustering
            min_cluster_size: Minimum chunks per cluster
        """
        self.embedding_model = embedding_model
        self.max_levels = max_levels
        
        # Initialize components
        self.clusterer = RAPTORClusterer(
            reduction_dimension=reduction_dimension,
            min_cluster_size=min_cluster_size
        )
        self.summarizer = RAPTORSummarizer()
        
        # Tree structure
        self.nodes: List[RAPTORNode] = []
        self.levels: Dict[int, List[int]] = {}  # level → node indices
        
        logger.info(f"✅ RAPTOR Tree initialized (max_levels={max_levels})")
    
    def build_tree(self, texts: List[str]) -> None:
        """
        Build RAPTOR tree from documents
        
        Args:
            texts: List of document chunks
        """
        logger.info(f"🔧 Building RAPTOR tree from {len(texts)} documents...")
        
        # Level 0: Original chunks (leaf nodes)
        logger.info("Level 0: Creating leaf nodes...")
        embeddings = self._embed_texts(texts)
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            node = RAPTORNode(
                text=text,
                embedding=embedding,
                level=0,
                is_summary=False
            )
            self.nodes.append(node)
        
        self.levels[0] = list(range(len(texts)))
        logger.info(f"  → {len(texts)} leaf nodes created")
        
        # Build higher levels
        current_level = 0
        current_texts = texts
        current_embeddings = embeddings
        
        while current_level < self.max_levels - 1:
            if len(current_texts) < self.clusterer.min_cluster_size:
                logger.info(f"Stopping at level {current_level}: too few nodes")
                break
            
            # Cluster current level
            logger.info(f"Level {current_level + 1}: Clustering...")
            clusters = self.clusterer.cluster_embeddings(
                current_embeddings,
                current_texts
            )
            
            if len(clusters) <= 1:
                logger.info(f"Stopping at level {current_level}: only one cluster")
                break
            
            # Summarize each cluster
            logger.info(f"Level {current_level + 1}: Summarizing {len(clusters)} clusters...")
            summaries = []
            summary_nodes = []
            
            for cluster_id, indices in clusters.items():
                # Get texts from current level
                cluster_texts = [current_texts[i] for i in indices]
                
                # Summarize
                summary = self.summarizer.summarize_cluster(cluster_texts, cluster_id)
                summaries.append(summary)
                
                # Create summary node
                summary_embedding = self._embed_texts([summary])[0]
                
                # Get actual node indices from current level
                if current_level == 0:
                    child_indices = indices
                else:
                    child_indices = [self.levels[current_level][i] for i in indices]
                
                node = RAPTORNode(
                    text=summary,
                    embedding=summary_embedding,
                    level=current_level + 1,
                    children=child_indices,
                    cluster_id=cluster_id,
                    is_summary=True
                )
                
                node_idx = len(self.nodes)
                self.nodes.append(node)
                summary_nodes.append(node_idx)
            
            # Update for next level
            self.levels[current_level + 1] = summary_nodes
            current_texts = summaries
            current_embeddings = np.array([self.nodes[i].embedding for i in summary_nodes])
            current_level += 1
            
            logger.info(f"  → {len(summaries)} summary nodes created")
        
        logger.info(f"✅ RAPTOR tree built with {len(self.levels)} levels, {len(self.nodes)} total nodes")
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using embedding model
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings
        """
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings)
    
    def retrieve_from_tree(
        self,
        query: str,
        top_k: int = 5,
        search_level: Optional[int] = None,
        collapse_tree: bool = True
    ) -> List[Tuple[str, float, int]]:
        """
        Retrieve from RAPTOR tree
        
        Args:
            query: Search query
            top_k: Number of results
            search_level: Specific level to search (None = all levels)
            collapse_tree: Search all levels and combine
            
        Returns:
            List of (text, score, level) tuples
        """
        # Embed query
        query_embedding = self._embed_texts([query])[0]
        
        # Determine which levels to search
        if search_level is not None:
            levels_to_search = [search_level] if search_level in self.levels else []
        elif collapse_tree:
            levels_to_search = list(self.levels.keys())
        else:
            levels_to_search = [0]  # Only leaf level
        
        # Search each level
        all_results = []
        
        for level in levels_to_search:
            node_indices = self.levels[level]
            
            for node_idx in node_indices:
                node = self.nodes[node_idx]
                
                # Calculate similarity (cosine)
                similarity = np.dot(query_embedding, node.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
                )
                
                all_results.append((node.text, float(similarity), node.level))
        
        # Sort by similarity (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return all_results[:top_k]
    
    def get_tree_stats(self) -> Dict:
        """
        Get tree statistics
        
        Returns:
            Dict with tree stats
        """
        stats = {
            "total_nodes": len(self.nodes),
            "num_levels": len(self.levels),
            "nodes_per_level": {
                level: len(indices) for level, indices in self.levels.items()
            },
            "leaf_nodes": len(self.levels.get(0, [])),
            "summary_nodes": sum(1 for node in self.nodes if node.is_summary)
        }
        
        return stats


def create_raptor_tree(
    texts: List[str],
    embedding_model,
    max_levels: int = 3
) -> RAPTORTree:
    """
    Factory function to create and build RAPTOR tree
    
    Args:
        texts: Document chunks
        embedding_model: Embedding model
        max_levels: Maximum tree depth
        
    Returns:
        Built RAPTORTree instance
    """
    tree = RAPTORTree(
        embedding_model=embedding_model,
        max_levels=max_levels
    )
    
    tree.build_tree(texts)
    
    return tree
