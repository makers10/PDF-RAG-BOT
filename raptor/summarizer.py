# raptor/summarizer.py
"""
Summarization for RAPTOR tree building
- Summarizes clusters of chunks
- Creates higher-level abstractions
- Uses FLAN-T5 for summarization
"""

import logging
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class RAPTORSummarizer:
    """
    Summarizer for RAPTOR tree construction
    
    How it works:
    1. Takes a cluster of chunks
    2. Combines them into context
    3. Generates abstractive summary
    4. Summary becomes parent node
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_length: int = 1024,
        max_summary_length: int = 256
    ):
        """
        Initialize summarizer
        
        Args:
            model_name: HuggingFace model for summarization
            max_input_length: Max tokens for input
            max_summary_length: Max tokens for summary
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        
        try:
            logger.info(f"🔧 Loading summarizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.summarizer = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU
            )
            
            logger.info(f"✅ Summarizer loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load summarizer: {e}")
            self.summarizer = None
    
    def summarize_cluster(
        self,
        texts: List[str],
        cluster_id: int = 0
    ) -> str:
        """
        Summarize a cluster of texts
        
        Args:
            texts: List of text chunks to summarize
            cluster_id: Cluster identifier
            
        Returns:
            Summary text
        """
        if not self.summarizer or not texts:
            return " ".join(texts[:3])  # Fallback: concatenate first 3
        
        try:
            # Combine texts
            combined_text = "\n\n".join(texts)
            
            # Truncate if too long
            tokens = self.tokenizer.encode(combined_text, add_special_tokens=False)
            if len(tokens) > self.max_input_length:
                tokens = tokens[:self.max_input_length]
                combined_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Create summarization prompt
            prompt = f"""Summarize the following text passages into a coherent, comprehensive summary. 
Capture the main ideas and key information.

Text:
{combined_text}

Summary:"""
            
            # Generate summary
            result = self.summarizer(
                prompt,
                max_new_tokens=self.max_summary_length,
                min_new_tokens=50,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            summary = result[0]["generated_text"].strip()
            
            logger.debug(f"Cluster {cluster_id}: {len(texts)} texts → {len(summary)} chars summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return first few texts
            return " ".join(texts[:3])
    
    def summarize_multiple_clusters(
        self,
        clusters: dict,
        texts: List[str]
    ) -> List[str]:
        """
        Summarize multiple clusters
        
        Args:
            clusters: Dict mapping cluster_id → list of indices
            texts: Original texts
            
        Returns:
            List of summaries (one per cluster)
        """
        summaries = []
        
        for cluster_id, indices in clusters.items():
            cluster_texts = [texts[i] for i in indices]
            summary = self.summarize_cluster(cluster_texts, cluster_id)
            summaries.append(summary)
        
        logger.info(f"✅ Summarized {len(clusters)} clusters")
        
        return summaries


def create_summarizer(
    model_name: str = "google/flan-t5-base"
) -> RAPTORSummarizer:
    """
    Factory function to create summarizer
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        RAPTORSummarizer instance
    """
    return RAPTORSummarizer(model_name)
