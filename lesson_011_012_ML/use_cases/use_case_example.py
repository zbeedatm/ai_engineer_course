import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from pydantic import BaseModel, Field
from google import genai

load_dotenv()

# =========================
# ✅ SETTINGS
# =========================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY_2") or os.environ.get("GEMINI_API_KEY")

# ✅ Input CSV path (use the CSV you already generated)
INPUT_CSV_PATH = "./lovable_client_messages_10000_nodash.csv"

# ✅ Column in CSV that contains the client message text
# In the files we generated earlier, it's "text"
TEXT_COL = "text"

# Optional metric columns. If missing, script will generate stand-ins.
VISITOR_COL = "visitor_id"
ACCOUNT_COL = "account_id"
SESSION_COL = "session_id"
TS_COL = "timestamp"  # if your CSV doesn't have timestamps, leave it; it will be ignored.

# Output folder
OUTDIR = "./out_usecases_from_csv"

# Gemini models
LABEL_MODEL = "gemini-2.5-flash"        # for naming clusters
EMBED_MODEL = "gemini-embedding-001"    # for embeddings

# Embedding batching
EMBED_BATCH_SIZE = 100

# Clustering
USE_MINIBATCH = False      # True for large datasets (100k+)
K_FIXED = 0                # 0 means "auto-select K"
K_MIN = 8
K_MAX = 30
K_SEARCH_SAMPLE_SIZE = 5000

# Cluster labeling
REPRESENTATIVES_PER_CLUSTER = 15

# Time window (applies only if TS_COL exists and is parseable)
LAST_N_DAYS = 90

# Rage detection (simple heuristic; can be upgraded to Gemini classification later)
RAGE_HINTS = [
    "stupid", "idiot", "wtf", "what the hell", "bullshit", "useless", "trash", "garbage",
    "fuck", "shit", "damn", "hate", "angry"
]


# =========================
# Gemini structured JSON schema
# =========================
class UseCaseLabel(BaseModel):
    use_case: str = Field(description="Short title (3-7 words).")
    description: str = Field(description="One sentence description (<= 25 words).")
    keywords: List[str] = Field(description="3-8 keywords/phrases.")


# =========================
# Helpers
# =========================
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
    """
    If your CSV doesn't include visitor/account/session IDs yet,
    we generate stand-ins so the output table still has those columns.
    Replace these with your real data when ready.
    """
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


def _extract_vec(e: Any) -> List[float]:
    # Robust extraction for SDK response shapes
    if e is None:
        return []
    if isinstance(e, dict):
        if "values" in e:
            return e["values"]
        if "embedding" in e and isinstance(e["embedding"], dict) and "values" in e["embedding"]:
            return e["embedding"]["values"]
    if hasattr(e, "values"):
        return list(getattr(e, "values"))
    if hasattr(e, "embedding") and hasattr(getattr(e, "embedding"), "values"):
        return list(e.embedding.values)
    return list(e)


def embed_texts_gemini(client: genai.Client, texts: List[str]) -> np.ndarray:
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        res = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        all_vecs.extend([_extract_vec(e) for e in res.embeddings])

    vecs = np.asarray(all_vecs, dtype=np.float32)
    vecs = normalize(vecs, norm="l2")  # cosine-friendly
    return vecs


def choose_k(vecs: np.ndarray) -> Tuple[int, Dict[int, float]]:
    """
    Choose K via silhouette score (cosine). Uses a sample for speed.
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

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine")
        scores[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k

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
    (Hard-coded simple definition; update later if you want per-session or per-day retention.)
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


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Hard-coded auth
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    client = genai.Client()

    # 1) Read CSV
    df = pd.read_csv(INPUT_CSV_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(f"CSV missing TEXT_COL='{TEXT_COL}'. Columns: {list(df.columns)}")

    # 2) Cleanup and optional window filter
    df = ensure_metric_cols(df)
    df = try_filter_last_n_days(df)

    df[TEXT_COL] = df[TEXT_COL].map(clean_text)
    df = df[df[TEXT_COL].str.len() >= 3].copy()

    df["is_rage"] = df[TEXT_COL].map(is_rage_prompt).astype(bool)

    texts = df[TEXT_COL].tolist()

    # 3) Gemini embeddings
    vecs = embed_texts_gemini(client, texts)

    # 4) Choose K (or use fixed)
    if K_FIXED and K_FIXED > 0:
        k = K_FIXED
        k_scores = {}
    else:
        k, k_scores = choose_k(vecs)

    # 5) Cluster
    labels, centers = cluster_vecs(vecs, k)
    df["cluster_id"] = labels

    # 6) Pick representatives + label clusters using Gemini
    reps = representative_indices(vecs, labels, centers)
    use_case_map: Dict[int, Dict[str, Any]] = {}

    for cid, idxs in reps.items():
        examples = df.iloc[idxs][TEXT_COL].tolist()
        use_case_map[cid] = label_cluster_with_gemini(client, examples)

    # Attach labels to rows
    df["Use Case"] = df["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("use_case", "Unknown"))
    df["Description"] = df["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("description", ""))

    # 7) Build your “Top use cases” table
    use_cases = build_use_case_table(df)
    use_cases["Use Case"] = use_cases["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("use_case", "Unknown"))
    use_cases["Description"] = use_cases["cluster_id"].map(lambda c: use_case_map.get(int(c), {}).get("description", ""))

    # Reorder like your screenshot
    use_cases = use_cases[
        ["Use Case", "Description", "Prompts", "Visitors", "Accounts", "Retention", "% Rage Prompts", "cluster_id"]
    ]

    # 8) Save outputs
    use_cases.to_csv(os.path.join(OUTDIR, "use_cases.csv"), index=False)
    df.to_csv(os.path.join(OUTDIR, "labeled_rows.csv"), index=False)
    with open(os.path.join(OUTDIR, "use_case_map.json"), "w", encoding="utf-8") as f:
        json.dump(use_case_map, f, ensure_ascii=False, indent=2)

    if k_scores:
        with open(os.path.join(OUTDIR, "k_silhouette_scores.json"), "w", encoding="utf-8") as f:
            json.dump(k_scores, f, indent=2)

    print(f"\nDone. Outputs written to: {OUTDIR}")
    print(f"Chosen K = {k}")
    print("\nTop 10 use cases:")
    print(use_cases.head(10).to_string(index=False))


if __name__ == "__main__":
    main()