from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Using BGE-Large for state-of-the-art retrieval performance.
    
    BGE-Small-EN-v1.5:
    - 384 dimensions (Memory efficient)
    - Good performance on MTEB benchmark
    - Optimized for retrieval tasks
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # Important for cosine similarity
            "batch_size": 32  # Process multiple texts at once
        }
    )

    return embeddings
