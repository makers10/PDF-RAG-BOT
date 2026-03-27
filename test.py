from rag_pipeline import create_vectorstore_from_pdf, answer_question

# Change this to your actual PDF filename
pdf_path = "sample.pdf"

print("Creating vector store...")
vector_store, multi_level_retriever, raptor_tree = create_vectorstore_from_pdf(pdf_path)

while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    print("\nSearching for answer...")
    answer = answer_question(
        vector_store, 
        query, 
        multi_level_retriever=multi_level_retriever,
        raptor_tree=raptor_tree
    )

    print("\nAnswer:\n", answer)
