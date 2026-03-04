import chromadb

# This client automatically handles the v2 API pathing
client = chromadb.HttpClient(host='aiserver.anudina.com', port=8000)

print(client.heartbeat())
collection = client.get_or_create_collection(name="my_first_collection")

# 3. Add some test data
collection.add(
    documents=[
        "This is a document about Red Hat Linux and Docker.",
        "ChromaDB is a great vector database for AI apps.",
        "PostgreSQL is a powerful relational database."
    ],
    metadatas=[{"source": "linux_info"}, {"source": "ai_info"}, {"source": "db_info"}],
    ids=["id1", "id2", "id3"]
)

print("Success! Created 'my_first_collection' with 3 documents.")
print(f"Heartbeat: {client.heartbeat()}")