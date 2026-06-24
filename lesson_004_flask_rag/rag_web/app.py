import os
import threading
import faiss
import numpy as np
import nltk
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------
# Config
# ---------------------------
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# ---------------------------
# Flask + globals
# ---------------------------
app = Flask(__name__)
os.makedirs(DATA_DIR, exist_ok=True)

_lock = threading.Lock()
_sentences = []
_index = None
_embed_model = None
_gemini_client = None

# ---------------------------
# NLTK setup (robust across versions)
# ---------------------------
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()
_stop_words = set(stopwords.words("english"))

# ---------------------------
# RAG functions
# ---------------------------
def load_documents(folder=DATA_DIR):
    all_sentences = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                for sentence in sent_tokenize(text):
                    words = word_tokenize(sentence)
                    clean = [w for w in words if w.lower() not in _stop_words and w.isalnum()]
                    if clean:
                        all_sentences.append(" ".join(clean))
    return all_sentences

def create_faiss(sentences, model):
    if not sentences:
        return None
    emb = model.encode(sentences)
    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index

def retrieve(query, model, index, sentences, k=3):
    if not sentences or index is None:
        return []
    q_emb = model.encode([query]).astype("float32")
    k = min(k, len(sentences))
    D, I = index.search(q_emb, k)
    return [sentences[i] for i in I[0]]

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

def ask_gemini(context, question):
    client = get_gemini_client()
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1)  # keep it fast/light
        )
    )
    return (resp.text or "").strip()

def rebuild_index():
    global _sentences, _index, _embed_model
    with _lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        _sentences = load_documents()
        _index = create_faiss(_sentences, _embed_model)

# ---------------------------
# Pages
# ---------------------------
@app.get("/")
def home():
    return render_template("index.html")

# ---------------------------
# APIs
# ---------------------------
@app.get("/health")
def health():
    with _lock:
        return jsonify({"status": "ok", "docs": len(_sentences), "indexed": _index is not None})

@app.post("/upload")
def upload():
    if "files" not in request.files:
        return jsonify({"error": "Send file(s) under 'files' form field."}), 400
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if f.filename and f.filename.lower().endswith(".txt"):
            out_path = os.path.join(DATA_DIR, os.path.basename(f.filename))
            f.save(out_path)
            saved.append(os.path.basename(out_path))
    rebuild_index()
    with _lock:
        return jsonify({"saved": saved, "docs_indexed": len(_sentences)})

@app.post("/reindex")
def reindex():
    rebuild_index()
    with _lock:
        return jsonify({"docs_indexed": len(_sentences), "indexed": _index is not None})

@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    k = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "Provide 'question' in JSON body."}), 400
    with _lock:
        if not _sentences or _index is None:
            return jsonify({"error": "No documents indexed. Upload or reindex first."}), 400
        chunks = retrieve(question, _embed_model, _index, _sentences, k=k)
    answer = ask_gemini("\n".join(chunks), question)
    return jsonify({"question": question, "top_k": k, "context": chunks, "answer": answer})

# ---------------------------
# Start
# ---------------------------
if __name__ == "__main__":
    rebuild_index()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
