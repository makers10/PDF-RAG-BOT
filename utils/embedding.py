from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )

    return embeddings
