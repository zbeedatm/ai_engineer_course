import os
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, render_template, abort
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from pydantic import BaseModel, Field
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# ✅ SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY_2") or os.environ.get("GEMINI_API_KEY", "")).strip()

# Default CSV in the same folder as app.py (optional)
DEFAULT_CSV_PATH = str(BASE_DIR / "lovable_client_messages_10000_nodash.csv")

TEXT_COL = "text"

VISITOR_COL = "visitor_id"
ACCOUNT_COL = "account_id"
SESSION_COL = "session_id"
TS_COL = "timestamp"

OUT_ROOT = str(BASE_DIR / "out_usecases_flask_fe")

# Gemini text model for cluster naming
LABEL_MODEL = "gemini-2.5-flash"

# Hugging Face embedding model (local, no rate limits)
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# If you need multilingual (Hebrew/Russian etc.), use:
# HF_MODEL_NAME = "intfloat/multilingual-e5-small"

# Clustering
USE_MINIBATCH = False
K_FIXED = 0                  # 0 = auto
K_MIN = 8
K_MAX = 30
K_SEARCH_SAMPLE_SIZE = 5000

REPRESENTATIVES_PER_CLUSTER = 15

LAST_N_DAYS = 90

RAGE_HINTS = [
    "stupid", "idiot", "wtf", "what the hell", "bullshit", "useless", "trash", "garbage",
    "fuck", "shit", "damn", "hate", "angry"
]


# ============================================================
# Flask app + background jobs
# ============================================================

app = Flask(__name__)
os.makedirs(OUT_ROOT, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

_hf_model = None
_hf_model_lock = threading.Lock()


# ============================================================
# Pydantic schema for Gemini JSON response
# ============================================================

class UseCaseLabel(BaseModel):
    use_case: str = Field(description="Short title (3-7 words).")
    description: str = Field(description="One sentence description (<= 25 words).")
    keywords: List[str] = Field(description="3-8 keywords/phrases.")


# ============================================================
# Helper functions
# ============================================================

def _job_update(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def clean_text(x: Any) -> str:
    if not isinstance(x, str):
        return ""
    return " ".join(x.split()).strip()


def is_rage_prompt(text: str) -> bool:
    t = text.lower()
    if any(h in t for h in RAGE_HINTS):
        return True
    if text.count("!") >= 3 or text.count("?") >= 4:
        return True
    letters = [c for c in text if c.isalpha()]
    if letters:
        caps = sum(1 for c in letters if c.isupper())
        if caps / len(letters) > 0.6 and len(letters) > 15:
            return True
    return False


def ensure_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    if VISITOR_COL not in df.columns:
        df[VISITOR_COL] = df.index.map(lambda i: f"v{i % 2500}")
    if ACCOUNT_COL not in df.columns:
        df[ACCOUNT_COL] = df.index.map(lambda i: f"a{i % 500}")
    if SESSION_COL not in df.columns:
        df[SESSION_COL] = df.index.map(lambda i: f"s{i % 4000}")
    return df


def try_filter_last_n_days(df: pd.DataFrame) -> pd.DataFrame:
    if TS_COL not in df.columns:
        return df
    ts = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
    if ts.notna().sum() == 0:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(days=LAST_N_DAYS)
    return df.loc[ts >= cutoff].copy()


def load_hf_model(job_id: str) -> SentenceTransformer:
    global _hf_model
    with _hf_model_lock:
        if _hf_model is None:
            _job_update(job_id, detail=f"Loading HF model: {HF_MODEL_NAME}", progress=3)
            _hf_model = SentenceTransformer(HF_MODEL_NAME)
        return _hf_model


def embed_texts_hf(texts: List[str], job_id: str) -> np.ndarray:
    """
    Local embeddings, no rate limits.
    """
    model = load_hf_model(job_id)

    texts = [t[:4000] for t in texts]  # avoid extreme lengths
    _job_update(job_id, detail="Embedding with Hugging Face (local)", progress=5)

    vecs = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    vecs = normalize(vecs, norm="l2")
    _job_update(job_id, detail="HF embeddings done", progress=55)
    return vecs


def choose_k(vecs: np.ndarray, job_id: str) -> Tuple[int, Dict[int, float]]:
    """
    Choose K via silhouette score (cosine) on a sample for speed.
    """
    scores: Dict[int, float] = {}
    n = vecs.shape[0]

    if n <= K_SEARCH_SAMPLE_SIZE:
        X = vecs
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=K_SEARCH_SAMPLE_SIZE, replace=False)
        X = vecs[idx]

    best_k = K_MIN
    best_score = -1.0

    total_ks = (K_MAX - K_MIN + 1)
    done = 0

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            done += 1
            continue

        score = silhouette_score(X, labels, metric="cosine")
        scores[k] = float(score)

        if score > best_score:
            best_score = score
            best_k = k

        done += 1
        pct = 55 + int(done / total_ks * 15)
        _job_update(job_id, detail=f"Choosing K: tried k={k}", progress=pct)

    return best_k, scores


def cluster_vecs(vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if USE_MINIBATCH:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")

    labels = km.fit_predict(vecs).astype(int)
    centers = km.cluster_centers_
    return labels, centers


def representative_indices(vecs: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Dict[int, List[int]]:
    reps: Dict[int, List[int]] = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        c = centers[cid]
        sims = vecs[idx] @ c
        order = np.argsort(-sims)
        reps[int(cid)] = idx[order[:REPRESENTATIVES_PER_CLUSTER]].tolist()
    return reps


def label_cluster_with_gemini(client: genai.Client, examples: List[str]) -> Dict[str, Any]:
    prompt = (
        "You are a product analytics expert.\n"
        "These are example user messages from ONE cluster.\n"
        "Return JSON with:\n"
        "- use_case: short title (3-7 words)\n"
        "- description: 1 sentence (<= 25 words)\n"
        "- keywords: 3-8 keywords/phrases\n\n"
        "Examples:\n" + "\n".join(f"- {e}" for e in examples)
    )

    resp = client.models.generate_content(
        model=LABEL_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": UseCaseLabel.model_json_schema(),
            "temperature": 0.2,
        },
    )

    try:
        return json.loads(resp.text)
    except Exception:
        return {"use_case": "Unknown", "description": (resp.text or "")[:240], "keywords": []}


def build_use_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retention proxy: % of visitors with >=2 messages inside that use case.
    """
    rows = []
    for cid, sub in df.groupby("cluster_id"):
        prompts = len(sub)
        visitors = sub[VISITOR_COL].nunique()
        accounts = sub[ACCOUNT_COL].nunique()

        by_visitor = sub.groupby(VISITOR_COL).size()
        retention = float((by_visitor >= 2).mean()) if len(by_visitor) else 0.0

        rage_pct = float(sub["is_rage"].mean()) if len(sub) else 0.0

        rows.append({
            "cluster_id": int(cid),
            "Prompts": int(prompts),
            "Visitors": int(visitors),
            "Accounts": int(accounts),
            "Retention": round(retention, 3),
            "% Rage Prompts": round(rage_pct * 100.0, 1),
        })

    return pd.DataFrame(rows).sort_values("Prompts", ascending=False)


# ============================================================
# Pipeline runner (background thread)
# ============================================================

def run_pipeline_job(job_id: str, input_csv_path: str, outdir: str):
    try:
        _job_update(job_id, status="running", progress=1, detail="Starting")

        # 1) Read CSV
        _job_update(job_id, progress=2, detail="Reading CSV")
        df = pd.read_csv(input_csv_path)

        if TEXT_COL not in df.columns:
            raise ValueError(f"CSV missing '{TEXT_COL}' column. Found: {list(df.columns)}")

        # 2) Cleanup + optional date filter
        df = ensure_metric_cols(df)
        df = try_filter_last_n_days(df)
        df[TEXT_COL] = df[TEXT_COL].map(clean_text)
        df = df[df[TEXT_COL].str.len() >= 3].copy()
        df["is_rage"] = df[TEXT_COL].map(is_rage_prompt).astype(bool)

        texts = df[TEXT_COL].tolist()
        if len(texts) < 20:
            raise ValueError("Not enough rows after filtering/cleanup (need at least ~20).")

        # 3) HF embeddings
        vecs = embed_texts_hf(texts, job_id)

        # 4) Choose K (or fixed)
        _job_update(job_id, progress=56, detail="Selecting K")
        if K_FIXED and K_FIXED > 0:
            k = K_FIXED
            k_scores: Dict[int, float] = {}
        else:
            k, k_scores = choose_k(vecs, job_id)

        # 5) Cluster
        _job_update(job_id, progress=72, detail=f"Clustering (k={k})")
        labels, centers = cluster_vecs(vecs, k)
        df["cluster_id"] = labels

        # 6) Representatives
        _job_update(job_id, progress=78, detail="Picking representatives")
        reps = representative_indices(vecs, labels, centers)

        # 7) Label clusters with Gemini
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY (required for cluster labeling).")

        client = genai.Client(api_key=GEMINI_API_KEY)

        _job_update(job_id, progress=82, detail="Labeling clusters with Gemini")
        use_case_map: Dict[int, Dict[str, Any]] = {}

        cids = sorted(reps.keys())
        total = len(cids)
        for i, cid in enumerate(cids, start=1):
            examples = df.iloc[reps[cid]][TEXT_COL].tolist()
            use_case_map[cid] = label_cluster_with_gemini(client, examples)

            pct = 82 + int(i / total * 10)
            _job_update(job_id, progress=pct, detail=f"Labeled cluster {i}/{total}")

        # Attach labels to rows
        df["Use Case"] = df["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("use_case", "Unknown"))
        df["Description"] = df["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("description", ""))

        # 8) Build table
        _job_update(job_id, progress=95, detail="Building use cases table")
        use_cases = build_use_case_table(df)
        use_cases["Use Case"] = use_cases["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("use_case", "Unknown"))
        use_cases["Description"] = use_cases["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("description", ""))

        use_cases = use_cases[
            ["Use Case", "Description", "Prompts", "Visitors", "Accounts", "Retention", "% Rage Prompts", "cluster_id"]
        ]

        # 9) Save outputs
        os.makedirs(outdir, exist_ok=True)

        use_cases_path = os.path.join(outdir, "use_cases.csv")
        labeled_rows_path = os.path.join(outdir, "labeled_rows.csv")
        use_case_map_path = os.path.join(outdir, "use_case_map.json")
        k_scores_path = os.path.join(outdir, "k_silhouette_scores.json")

        use_cases.to_csv(use_cases_path, index=False)
        df.to_csv(labeled_rows_path, index=False)
        with open(use_case_map_path, "w", encoding="utf-8") as f:
            json.dump(use_case_map, f, ensure_ascii=False, indent=2)

        if k_scores:
            with open(k_scores_path, "w", encoding="utf-8") as f:
                json.dump(k_scores, f, indent=2)

        preview = use_cases.head(30).to_dict(orient="records")

        _job_update(
            job_id,
            status="done",
            progress=100,
            detail="Done",
            result={
                "k": int(k),
                "rows": int(len(df)),
                "use_cases": int(len(use_cases)),
                "files": {
                    "use_cases_csv": "use_cases.csv",
                    "labeled_rows_csv": "labeled_rows.csv",
                    "use_case_map_json": "use_case_map.json",
                    "k_scores_json": "k_silhouette_scores.json" if k_scores else None,
                },
                "preview": preview,
            }
        )

    except Exception as e:
        # IMPORTANT: expose real error text
        _job_update(job_id, status="error", error=str(e), detail=str(e), progress=0)


# ============================================================
# Routes
# ============================================================

@app.get("/")
def index():
    return render_template(
        "index.html",
        text_col=TEXT_COL,
        default_csv_exists=os.path.isfile(DEFAULT_CSV_PATH),
        hf_model=HF_MODEL_NAME,
        label_model=LABEL_MODEL
    )


@app.post("/api/run")
def api_run():
    job_id = uuid.uuid4().hex[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUT_ROOT, f"run_{ts}_{job_id}")
    os.makedirs(outdir, exist_ok=True)

    # input selection
    input_path = DEFAULT_CSV_PATH

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        input_path = os.path.join(outdir, "input.csv")
        f.save(input_path)
    else:
        # If no file uploaded, ensure default exists
        if not os.path.isfile(DEFAULT_CSV_PATH):
            return jsonify({"error": f"Default CSV not found: {DEFAULT_CSV_PATH}. Upload a file instead."}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "detail": "Queued",
            "error": None,
            "result": None,
            "outdir": outdir,
        }

    t = threading.Thread(target=run_pipeline_job, args=(job_id, input_path, outdir), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.get("/api/status/<job_id>")
def api_status(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            return jsonify({"error": "Unknown job_id"}), 404
        payload = JOBS[job_id].copy()

    payload.pop("outdir", None)
    return jsonify(payload)


def _job_outdir(job_id: str) -> str:
    with JOBS_LOCK:
        if job_id not in JOBS:
            abort(404, description="Unknown job_id")
        return JOBS[job_id]["outdir"]


@app.get("/runs/<job_id>/<path:filename>")
def download(job_id: str, filename: str):
    allowed = {"use_cases.csv", "labeled_rows.csv", "use_case_map.json", "k_silhouette_scores.json"}
    if filename not in allowed:
        abort(404)

    outdir = _job_outdir(job_id)
    path = os.path.join(outdir, filename)
    if not os.path.isfile(path):
        abort(404)

    mimetype = "text/csv" if filename.endswith(".csv") else "application/json"
    return send_file(path, mimetype=mimetype, as_attachment=True, download_name=filename)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "default_csv_exists": os.path.isfile(DEFAULT_CSV_PATH),
        "hf_model": HF_MODEL_NAME,
        "label_model": LABEL_MODEL
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
