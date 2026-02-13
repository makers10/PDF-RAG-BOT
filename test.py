from rag_pipeline import create_vectorstore_from_pdf, create_qa_chain

# Change this to your actual PDF filename
pdf_path = "sample.pdf"

print("Creating vector store...")
vector_store = create_vectorstore_from_pdf(pdf_path)

print("Creating QA chain...")
qa_chain = create_qa_chain(vector_store)

while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    response = qa_chain(query)

    print("\nAnswer:\n", response["result"])

    print("\nSources:\n")
    for doc in response["source_documents"]:
        print(doc.page_content[:300])
        print("------")
