from app.documents.loader import load_document, prepare_documents
from app.embeddings.voyage import create_embeddings


document = load_document("data/data.json")

prepared_documents = prepare_documents(document)

embeddings = create_embeddings(prepared_documents)

print("Number of documents:", len(prepared_documents))
print("Number of embeddings:", len(embeddings))
print("Vector size:", len(embeddings[0]))
print("First 10 values:", embeddings[0][:10])