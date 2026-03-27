# cache/redis_cache.py
import redis
import pickle
import hashlib
import logging
from typing import Optional, Any, List
from config.redis_config import REDIS_CONFIG, CACHE_TTL, CACHE_PREFIX

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis caching layer for RAG system
    - Caches embeddings, QA pairs, and vector stores
    - TTL: 24 hours (86400 seconds)
    - Automatic key generation with MD5 hashing
    """
    
    def __init__(self, ttl: int = 86400):
        """
        Initialize Redis connection
        
        Args:
            ttl: Time-to-live in seconds (default: 24 hours)
        """
        self.ttl = ttl
        self.connected = False
        
        try:
            self.client = redis.Redis(**REDIS_CONFIG)
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info("✅ Redis connected successfully")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.warning("Cache will be disabled. Install Redis or use mock cache.")
            self.client = None
            self.connected = False
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """
        Generate cache key from data hash
        
        Args:
            prefix: Key prefix (e.g., 'qa', 'emb')
            data: Data to hash
            
        Returns:
            Cache key in format: prefix:hash
        """
        hash_obj = hashlib.md5(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.connected:
            return None
        
        try:
            data = self.client.get(key)
            if data:
                logger.debug(f"✅ Cache HIT: {key}")
                return pickle.loads(data)
            logger.debug(f"❌ Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cached value with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
        """
        if not self.connected:
            return
        
        try:
            ttl = ttl or self.ttl
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"💾 Cached: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete cached value"""
        if not self.connected:
            return
        
        try:
            self.client.delete(key)
            logger.debug(f"🗑️ Deleted: {key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def clear_all(self):
        """Clear all cache (use with caution!)"""
        if not self.connected:
            return
        
        try:
            self.client.flushdb()
            logger.info("🗑️ All cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    # -------------------- Specialized Cache Methods --------------------
    
    def cache_embeddings(self, text: str, embeddings: List[float]) -> str:
        """
        Cache document embeddings
        
        Args:
            text: Original text
            embeddings: Embedding vectors
            
        Returns:
            Cache key
        """
        key = self._generate_key(CACHE_PREFIX["embeddings"], text)
        self.set(key, embeddings, ttl=CACHE_TTL["embeddings"])
        return key
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Get cached embeddings
        
        Args:
            text: Original text
            
        Returns:
            Cached embeddings or None
        """
        key = self._generate_key(CACHE_PREFIX["embeddings"], text)
        return self.get(key)
    
    def cache_answer(self, query: str, answer: str, metadata: Optional[dict] = None):
        """
        Cache QA pair
        
        Args:
            query: User question
            answer: Generated answer
            metadata: Optional metadata (sources, confidence, etc.)
        """
        key = self._generate_key(CACHE_PREFIX["qa"], query)
        value = {
            "answer": answer,
            "metadata": metadata or {}
        }
        self.set(key, value, ttl=CACHE_TTL["qa_pairs"])
    
    def get_answer(self, query: str) -> Optional[dict]:
        """
        Get cached answer
        
        Args:
            query: User question
            
        Returns:
            Cached answer dict or None
        """
        key = self._generate_key(CACHE_PREFIX["qa"], query)
        return self.get(key)
    
    def cache_chunks(self, pdf_name: str, chunks: List[str]):
        """
        Cache document chunks
        
        Args:
            pdf_name: PDF filename
            chunks: List of text chunks
        """
        key = self._generate_key(CACHE_PREFIX["chunks"], pdf_name)
        self.set(key, chunks, ttl=CACHE_TTL["default"])
    
    def get_chunks(self, pdf_name: str) -> Optional[List[str]]:
        """
        Get cached chunks
        
        Args:
            pdf_name: PDF filename
            
        Returns:
            Cached chunks or None
        """
        key = self._generate_key(CACHE_PREFIX["chunks"], pdf_name)
        return self.get(key)
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dict with cache stats
        """
        if not self.connected:
            return {"connected": False}
        
        try:
            info = self.client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"connected": True, "error": str(e)}
    
    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> str:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return "0%"
        rate = (hits / total) * 100
        return f"{rate:.2f}%"


# -------------------- Mock Cache for Development --------------------

class MockCache:
    """
    Mock cache for development when Redis is not available
    Uses in-memory dictionary (not persistent)
    """
    
    def __init__(self, ttl: int = 86400):
        self.cache = {}
        self.ttl = ttl
        self.connected = False
        logger.warning("⚠️ Using MockCache (in-memory only)")
    
    def _generate_key(self, prefix: str, data: str) -> str:
        hash_obj = hashlib.md5(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = value
    
    def delete(self, key: str):
        self.cache.pop(key, None)
    
    def clear_all(self):
        self.cache.clear()
    
    def cache_answer(self, query: str, answer: str, metadata: Optional[dict] = None):
        key = self._generate_key(CACHE_PREFIX["qa"], query)
        self.set(key, {"answer": answer, "metadata": metadata or {}})
    
    def get_answer(self, query: str) -> Optional[dict]:
        key = self._generate_key(CACHE_PREFIX["qa"], query)
        return self.get(key)
    
    def get_stats(self) -> dict:
        return {
            "connected": False,
            "type": "mock",
            "total_keys": len(self.cache)
        }


# -------------------- Cache Factory --------------------

def get_cache(use_redis: bool = True, ttl: int = 86400):
    """
    Get cache instance (Redis or Mock)
    
    Args:
        use_redis: Try to use Redis (falls back to Mock if unavailable)
        ttl: Time-to-live in seconds
        
    Returns:
        Cache instance
    """
    if use_redis:
        cache = RedisCache(ttl=ttl)
        if cache.connected:
            return cache
    
    # Fallback to mock cache
    return MockCache(ttl=ttl)
