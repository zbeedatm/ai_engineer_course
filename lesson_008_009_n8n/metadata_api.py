"""
Local Metadata API – COVID-19 Article Classifier
Endpoint: POST http://localhost:8080/api/metadata
Used by the n8n workflow to enrich Ollama-extracted article data
with categories, sensitivity, department tags, and priority scoring.
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json

app = Flask(__name__)

# ──────────────────────────────────────────────
# Classification Tables
# ──────────────────────────────────────────────

RESEARCH_TYPE_TO_CATEGORY = {
    "clinical study":       "Clinical Research",
    "laboratory research":  "Basic Science",
    "review":               "Literature Review",
    "epidemiology":         "Public Health & Epidemiology",
    "genomics":             "Genomics & Bioinformatics",
    "drug/vaccine trial":   "Therapeutics & Vaccines",
    "public health":        "Public Health & Epidemiology",
    "other":                "Uncategorized",
    "unknown":              "Uncategorized",
}

RESEARCH_TYPE_TO_DEPARTMENT = {
    "clinical study":       "Clinical Affairs",
    "laboratory research":  "R&D / Laboratory",
    "review":               "Academic Affairs",
    "epidemiology":         "Epidemiology Unit",
    "genomics":             "Genomics Lab",
    "drug/vaccine trial":   "Drug Development",
    "public health":        "Public Health Division",
    "other":                "General",
    "unknown":              "General",
}

SENSITIVITY_RULES = {
    "High":   "Confidential",
    "Medium": "Internal",
    "Low":    "Public",
}

HIGH_PRIORITY_PATHOGENS = {
    "sars-cov-2", "sars-cov2", "covid-19", "covid19", "covid",
    "mers-cov", "mers", "sars", "coronavirus", "ibv",
    "influenza", "h1n1", "h5n1", "ebola", "mpox",
}

HIGH_PRIORITY_KEYWORDS = {
    "mortality", "transmission", "outbreak", "pandemic",
    "vaccine", "treatment", "icu", "ventilator",
    "clinical trial", "drug resistance", "mutation", "variant",
    "infection", "hospitalization", "fatality", "sequencing",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def normalize_str(value) -> str:
    """Safely coerce any value to a clean stripped string."""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip()


def normalize_list(value) -> list:
    """
    Safely coerce any value to a list of strings.
    Handles: actual list, JSON string '["a","b"]', comma-separated string, single string.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        val = value.strip()
        if not val:
            return []
        # Try JSON array first
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v]
            except (json.JSONDecodeError, ValueError):
                pass
        # Fall back to comma-separated
        return [v.strip() for v in val.split(",") if v.strip()]
    return []


def fuzzy_match_research_type(raw: str) -> str:
    """
    Match Ollama's research_type output to our lookup keys,
    tolerating case, extra spaces, and slight wording differences.
    """
    normalized = raw.strip().lower()

    # Direct match first
    if normalized in RESEARCH_TYPE_TO_CATEGORY:
        return normalized

    # Partial / fuzzy match
    for key in RESEARCH_TYPE_TO_CATEGORY:
        if key in normalized or normalized in key:
            return key

    # Common synonyms Ollama might output
    synonyms = {
        "lab research":                "laboratory research",
        "basic research":              "laboratory research",
        "in vitro":                    "laboratory research",
        "in vivo":                     "laboratory research",
        "systematic review":           "review",
        "meta-analysis":               "review",
        "meta analysis":               "review",
        "epi study":                   "epidemiology",
        "epidemiological study":       "epidemiology",
        "rct":                         "drug/vaccine trial",
        "randomized controlled trial": "drug/vaccine trial",
        "vaccine trial":               "drug/vaccine trial",
        "drug trial":                  "drug/vaccine trial",
        "genomic":                     "genomics",
        "bioinformatics":              "genomics",
        "sequencing":                  "genomics",
        "public health study":         "public health",
        "health policy":               "public health",
    }
    for synonym, canonical in synonyms.items():
        if synonym in normalized:
            return canonical

    return "unknown"


def compute_priority_score(data: dict, pathogens: list, keywords: list) -> int:
    """Score 0-100 reflecting urgency. Higher = more critical."""
    score = 0

    # Clinical relevance (max 40 pts)
    relevance_pts = {"High": 40, "Medium": 20, "Low": 5}
    score += relevance_pts.get(data.get("clinical_relevance", "Low"), 0)

    # COVID-19 direct relevance (max 25 pts)
    covid_rel = data.get("covid19_relevance", "")
    if covid_rel == "Direct":
        score += 25
    elif covid_rel == "Indirect":
        score += 10

    # High-priority pathogens (max 20 pts)
    if any(p.lower() in HIGH_PRIORITY_PATHOGENS for p in pathogens):
        score += 20

    # High-priority keywords (max 15 pts, 5 pts each, capped)
    matched = sum(1 for k in keywords if k.lower() in HIGH_PRIORITY_KEYWORDS)
    score += min(matched * 5, 15)

    return min(score, 100)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/metadata", methods=["POST"])
def metadata():
    # Accept both application/json and raw body
    raw_body = request.get_data(as_text=True)
    try:
        data = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        data = request.get_json(force=True, silent=True) or {}

    if not data:
        return jsonify({"error": "Invalid or missing JSON body", "received": raw_body[:200]}), 400

    # ── Normalize all inputs ──────────────────
    research_type_raw  = normalize_str(data.get("research_type", "unknown"))
    clinical_relevance = normalize_str(data.get("clinical_relevance", "Low"))
    covid19_relevance  = normalize_str(data.get("covid19_relevance", "Not COVID-19"))
    pathogens          = normalize_list(data.get("pathogens_mentioned", []))
    keywords           = normalize_list(data.get("keywords", []))

    # ── Fuzzy-match research_type ─────────────
    matched_type = fuzzy_match_research_type(research_type_raw)

    # ── Derive fields ─────────────────────────
    category          = RESEARCH_TYPE_TO_CATEGORY.get(matched_type, "Uncategorized")
    department_tag    = RESEARCH_TYPE_TO_DEPARTMENT.get(matched_type, "General")
    sensitivity_level = SENSITIVITY_RULES.get(clinical_relevance, "Public")

    # Elevate sensitivity for direct COVID-19 hits
    if covid19_relevance == "Direct" and sensitivity_level == "Public":
        sensitivity_level = "Internal"

    review_required = (
        clinical_relevance == "High" or
        covid19_relevance == "Direct"
    )

    pathogens_flagged = [p for p in pathogens if p.lower() in HIGH_PRIORITY_PATHOGENS]
    priority_score    = compute_priority_score(data, pathogens, keywords)

    response = {
        "category":              category,
        "department_tag":        department_tag,
        "sensitivity_level":     sensitivity_level,
        "review_required":       review_required,
        "priority_score":        priority_score,
        "keywords":              keywords,
        "pathogens_flagged":     pathogens_flagged,
        "matched_research_type": matched_type,   # useful for debugging
        "processed_at":          datetime.utcnow().isoformat(),
    }

    return jsonify(response), 200


@app.route("/api/metadata", methods=["GET"])
def metadata_docs():
    return jsonify({
        "endpoint":    "POST /api/metadata",
        "port":        8080,
        "description": "Enriches COVID-19 article metadata from Ollama extraction",
        "expected_input": {
            "research_type":       "string (case-insensitive, fuzzy matched)",
            "covid19_relevance":   "string (Direct | Indirect | Not COVID-19)",
            "clinical_relevance":  "string (High | Medium | Low)",
            "pathogens_mentioned": "list[string] or JSON array string",
            "keywords":            "list[string] or JSON array string",
        },
        "returns": {
            "category":              "string",
            "department_tag":        "string",
            "sensitivity_level":     "string (Public | Internal | Confidential)",
            "review_required":       "boolean",
            "priority_score":        "integer (0-100)",
            "pathogens_flagged":     "list[string] — subset matching known high-risk pathogens",
            "keywords":              "list[string]",
            "matched_research_type": "string — normalized type used for lookup (debug)",
            "processed_at":          "ISO timestamp",
        }
    })


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("COVID-19 Metadata API running on http://localhost:8080")
    print("   POST /api/metadata  - classify an article")
    print("   GET  /api/metadata  - view schema docs")
    print("   GET  /health        - liveness check")
    app.run(host="localhost", port=8080, debug=False)
