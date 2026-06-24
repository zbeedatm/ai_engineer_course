import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

# -----------------------------
# 1. Connect to OpenSearch
# -----------------------------
host = os.environ.get("OPENSEARCH_HOST")
user = os.environ.get("OPENSEARCH_USER", "admin")
password = os.environ.get("OPENSEARCH_PASS")

client = OpenSearch(
    hosts=[host],
    http_auth=(user, password),
    use_ssl=True,
    verify_certs=True
)

# -----------------------------
# 2. Create an Index
# -----------------------------
index_name = "basic-index"

settings = {
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "content": {"type": "text"}
    }
  }
}

client.indices.create(index=index_name, body=settings, ignore=400)
print(f"Index '{index_name}' created (or already exists).")

# -----------------------------
# 3. Insert a Document
# # -----------------------------
# doc1 = {
#     "title": "RAG Intro",
#     "content": "Retrieval Augmented Generation connects search and AI."
# }
#
# response = client.index(index=index_name, body=doc1)
# print("Document inserted with ID:", response["_id"])

# -----------------------------
# 4. Search the Index
# -----------------------------
query = {
  "query": {
    "match": {
      "content": "and" #Search AI
    }
  }
}

results = client.search(index=index_name, body=query)

print("\nSearch results:")
for hit in results["hits"]["hits"]:
    print(hit["_source"])
