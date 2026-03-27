# raptor/clustering.py
"""
Clustering for RAPTOR tree building
- Groups similar chunks together
- Creates hierarchical structure
- Uses UMAP + GMM clustering
"""

import logging
import numpy as np
from typing import List, Tuple, Dict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import umap

logger = logging.getLogger(__name__)


class RAPTORClusterer:
    """
    Clustering for RAPTOR tree construction
    
    How it works:
    1. Reduce embedding dimensions with UMAP
    2. Cluster with Gaussian Mixture Model (GMM)
    3. Group similar chunks together
    4. Create hierarchical levels
    """
    
    def __init__(
        self,
        reduction_dimension: int = 10,
        n_neighbors: int = 15,
        min_cluster_size: int = 3,
        max_cluster_size: int = 10
    ):
        """
        Initialize clusterer
        
        Args:
            reduction_dimension: UMAP target dimensions
            n_neighbors: UMAP neighbors parameter
            min_cluster_size: Minimum chunks per cluster
            max_cluster_size: Maximum chunks per cluster
        """
        self.reduction_dimension = reduction_dimension
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        logger.info(f"✅ RAPTOR Clusterer initialized")
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        if len(embeddings) < self.n_neighbors:
            # Not enough samples for UMAP, use PCA fallback
            logger.warning(f"Too few samples ({len(embeddings)}), using mean pooling")
            return embeddings
        
        try:
            reducer = umap.UMAP(
                n_components=min(self.reduction_dimension, len(embeddings) - 1),
                n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
                metric='cosine',
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            logger.debug(f"Reduced {embeddings.shape} → {reduced.shape}")
            return reduced
        except Exception as e:
            logger.error(f"UMAP failed: {e}, using original embeddings")
            return embeddings
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str]
    ) -> Dict[int, List[int]]:
        """
        Cluster embeddings into groups
        
        Args:
            embeddings: Document embeddings
            texts: Original texts
            
        Returns:
            Dict mapping cluster_id → list of document indices
        """
        n_samples = len(embeddings)
        
        if n_samples < self.min_cluster_size:
            # Too few samples, return single cluster
            return {0: list(range(n_samples))}
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # Determine optimal number of clusters
        n_clusters = self._determine_n_clusters(n_samples)
        
        try:
            # Use GMM for soft clustering
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            labels = gmm.fit_predict(reduced_embeddings)
            
        except Exception as e:
            logger.warning(f"GMM failed: {e}, using KMeans")
            # Fallback to KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced_embeddings)
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Filter small clusters
        filtered_clusters = {}
        orphans = []
        
        for cluster_id, indices in clusters.items():
            if len(indices) >= self.min_cluster_size:
                filtered_clusters[cluster_id] = indices
            else:
                orphans.extend(indices)
        
        # Assign orphans to nearest cluster
        if orphans and filtered_clusters:
            for orphan_idx in orphans:
                # Find nearest cluster
                orphan_emb = reduced_embeddings[orphan_idx]
                min_dist = float('inf')
                nearest_cluster = list(filtered_clusters.keys())[0]
                
                for cluster_id, indices in filtered_clusters.items():
                    cluster_embs = reduced_embeddings[indices]
                    centroid = np.mean(cluster_embs, axis=0)
                    dist = np.linalg.norm(orphan_emb - centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cluster = cluster_id
                
                filtered_clusters[nearest_cluster].append(orphan_idx)
        elif orphans:
            # No valid clusters, create one
            filtered_clusters[0] = orphans
        
        logger.info(f"📊 Created {len(filtered_clusters)} clusters from {n_samples} documents")
        
        return filtered_clusters
    
    def _determine_n_clusters(self, n_samples: int) -> int:
        """
        Determine optimal number of clusters
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Optimal number of clusters
        """
        # Heuristic: aim for clusters of size 5-10
        target_cluster_size = (self.min_cluster_size + self.max_cluster_size) / 2
        n_clusters = max(2, int(n_samples / target_cluster_size))
        
        # Cap at reasonable maximum
        n_clusters = min(n_clusters, n_samples // self.min_cluster_size)
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        return n_clusters
    
    def hierarchical_clustering(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        max_levels: int = 3
    ) -> List[Dict[int, List[int]]]:
        """
        Create hierarchical clustering (multi-level)
        
        Args:
            embeddings: Document embeddings
            texts: Original texts
            max_levels: Maximum tree depth
            
        Returns:
            List of cluster dictionaries, one per level
        """
        levels = []
        current_embeddings = embeddings
        current_texts = texts
        
        for level in range(max_levels):
            if len(current_texts) < self.min_cluster_size:
                logger.info(f"Stopping at level {level}: too few documents")
                break
            
            # Cluster current level
            clusters = self.cluster_embeddings(current_embeddings, current_texts)
            levels.append(clusters)
            
            logger.info(f"Level {level}: {len(clusters)} clusters")
            
            # Stop if only one cluster
            if len(clusters) <= 1:
                break
            
            # Prepare for next level (would need summarization here)
            # For now, we'll stop
            break
        
        return levels


def create_clusterer(
    reduction_dimension: int = 10,
    min_cluster_size: int = 3
) -> RAPTORClusterer:
    """
    Factory function to create clusterer
    
    Args:
        reduction_dimension: UMAP dimensions
        min_cluster_size: Minimum cluster size
        
    Returns:
        RAPTORClusterer instance
    """
    return RAPTORClusterer(
        reduction_dimension=reduction_dimension,
        min_cluster_size=min_cluster_size
    )
