"""
Local Metadata API – COVID-19 Article Classifier
Endpoint: POST http://localhost:8080/api/metadata
Used by the n8n workflow to enrich Ollama-extracted article data
with categories, sensitivity, department tags, and priority scoring.
"""

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ──────────────────────────────────────────────
# Classification Rules
# ──────────────────────────────────────────────

RESEARCH_TYPE_TO_CATEGORY = {
    "Clinical Study":       "Clinical Research",
    "Laboratory Research":  "Basic Science",
    "Review":               "Literature Review",
    "Epidemiology":         "Public Health & Epidemiology",
    "Genomics":             "Genomics & Bioinformatics",
    "Drug/Vaccine Trial":   "Therapeutics & Vaccines",
    "Public Health":        "Public Health & Epidemiology",
    "Other":                "Uncategorized",
    "UNKNOWN":              "Uncategorized",
}

RESEARCH_TYPE_TO_DEPARTMENT = {
    "Clinical Study":       "Clinical Affairs",
    "Laboratory Research":  "R&D / Laboratory",
    "Review":               "Academic Affairs",
    "Epidemiology":         "Epidemiology Unit",
    "Genomics":             "Genomics Lab",
    "Drug/Vaccine Trial":   "Drug Development",
    "Public Health":        "Public Health Division",
    "Other":                "General",
    "UNKNOWN":              "General",
}

SENSITIVITY_RULES = {
    # clinical_relevance → base sensitivity
    "High":   "Confidential",
    "Medium": "Internal",
    "Low":    "Public",
}

HIGH_PRIORITY_PATHOGENS = {
    "sars-cov-2", "sars-cov2", "covid-19", "covid19",
    "mers-cov", "mers", "sars", "coronavirus",
}

HIGH_PRIORITY_KEYWORDS = {
    "mortality", "transmission", "outbreak", "pandemic",
    "vaccine", "treatment", "icu", "ventilator",
    "clinical trial", "drug resistance", "mutation", "variant",
}

# ──────────────────────────────────────────────
# Priority Scoring
# ──────────────────────────────────────────────

def compute_priority_score(data: dict) -> int:
    """
    Score 0–100 reflecting how urgent/important this article is.
    Higher = more critical for review.
    """
    score = 0

    # Clinical relevance (max 40 pts)
    relevance_pts = {"High": 40, "Medium": 20, "Low": 5}
    score += relevance_pts.get(data.get("clinical_relevance", "Low"), 0)

    # COVID-19 direct relevance (max 25 pts)
    covid_relevance = data.get("covid19_relevance", "")
    if covid_relevance == "Direct":
        score += 25
    elif covid_relevance == "Indirect":
        score += 10

    # Dangerous pathogens mentioned (max 20 pts)
    pathogens = [p.lower() for p in data.get("pathogens_mentioned", [])]
    if any(p in HIGH_PRIORITY_PATHOGENS for p in pathogens):
        score += 20

    # High-priority keywords (max 15 pts)
    keywords = [k.lower() for k in data.get("keywords", [])]
    matched_kw = sum(1 for k in keywords if k in HIGH_PRIORITY_KEYWORDS)
    score += min(matched_kw * 5, 15)

    return min(score, 100)  # cap at 100


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/metadata", methods=["POST"])
def metadata():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    research_type    = data.get("research_type", "UNKNOWN")
    clinical_relevance = data.get("clinical_relevance", "Low")
    covid19_relevance  = data.get("covid19_relevance", "Not COVID-19")
    pathogens          = data.get("pathogens_mentioned", [])
    keywords           = data.get("keywords", [])

    # Derive fields
    category         = RESEARCH_TYPE_TO_CATEGORY.get(research_type, "Uncategorized")
    department_tag   = RESEARCH_TYPE_TO_DEPARTMENT.get(research_type, "General")
    sensitivity_level = SENSITIVITY_RULES.get(clinical_relevance, "Public")

    # Elevate sensitivity if direct COVID-19 hit
    if covid19_relevance == "Direct" and sensitivity_level == "Public":
        sensitivity_level = "Internal"

    # Review required if high clinical relevance OR direct COVID-19
    review_required = (
        clinical_relevance == "High" or
        covid19_relevance == "Direct"
    )

    priority_score = compute_priority_score(data)

    response = {
        "category":          category,
        "department_tag":    department_tag,
        "sensitivity_level": sensitivity_level,
        "review_required":   review_required,
        "priority_score":    priority_score,
        "keywords":          keywords,          # pass-through, enriched if needed
        "pathogens_flagged": [
            p for p in pathogens
            if p.lower() in HIGH_PRIORITY_PATHOGENS
        ],
        "processed_at":      datetime.utcnow().isoformat(),
    }

    return jsonify(response), 200


@app.route("/api/metadata", methods=["GET"])
def metadata_docs():
    """Quick self-documentation endpoint."""
    return jsonify({
        "endpoint":  "POST /api/metadata",
        "port":      8080,
        "description": "Enriches COVID-19 article metadata from Ollama extraction",
        "expected_input": {
            "research_type":      "string",
            "covid19_relevance":  "string (Direct | Indirect | Not COVID-19)",
            "clinical_relevance": "string (High | Medium | Low)",
            "pathogens_mentioned":"list[string]",
            "keywords":           "list[string]",
        },
        "returns": {
            "category":          "string",
            "department_tag":    "string",
            "sensitivity_level": "string (Public | Internal | Confidential)",
            "review_required":   "boolean",
            "priority_score":    "integer (0–100)",
            "pathogens_flagged": "list[string]",
            "keywords":          "list[string]",
            "processed_at":      "ISO timestamp",
        }
    })


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 COVID-19 Metadata API running on http://localhost:8080")
    print("   POST /api/metadata  – classify an article")
    print("   GET  /api/metadata  – view docs")
    print("   GET  /health        – health check")
    app.run(host="localhost", port=8080, debug=False)
