from rag_pipeline import answer_question
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from utils.embedding import get_embedding_model

def mock_vector_store():
    # Create a simple mock vector store with some text
    text = """The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined together by Qin Shi Huang (220–206 BC), the first emperor of China. Little of the Qin wall remains. Later, many successive dynasties have built and maintained multiple stretches of border walls. The most well-known sections of the wall were built by the Ming dynasty (1368–1644)."""
    
    doc = Document(page_content=text, metadata={"chunk_id": 0})
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents([doc], embedding_model)
    return vector_store

def test_repro():
    vs = mock_vector_store()
    query = "When were the most well-known sections of the Great Wall built?"
    
    # We pass use_cache=False to avoid using previously cached answers
    answer = answer_question(vs, query, use_cache=False, use_raptor=False, use_multi_level=False)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    test_repro()
