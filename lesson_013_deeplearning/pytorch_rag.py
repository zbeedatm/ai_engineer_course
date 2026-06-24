import json
from sentence_transformers import SentenceTransformer

# SentenceTransformer uses PyTorch internally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # fast + popular

def embed_texts(texts):
    # Returns numpy array (n, dim)
    emb = model.encode(texts, normalize_embeddings=True)
    return emb

if __name__ == "__main__":
    docs = [
        {"id": "doc1", "text": "Kubernetes is a container orchestration system."},
        {"id": "doc2", "text": "PyTorch is a deep learning framework in Python."},
    ]
    vectors = embed_texts([d["text"] for d in docs])
    for d, v in zip(docs, vectors):
        print(json.dumps({"id": d["id"], "dim": int(v.shape[0]), "head": [float(x) for x in v[:5]]}))
