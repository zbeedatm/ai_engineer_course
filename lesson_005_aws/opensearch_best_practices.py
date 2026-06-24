import os
from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

load_dotenv()

# -----------------------------------
# 1) CONNECT
# -----------------------------------
HOST = os.environ.get("OPENSEARCH_HOST")
USER = os.environ.get("OPENSEARCH_USER", "admin")
PASS = os.environ.get("OPENSEARCH_PASS")

client = OpenSearch(
    hosts=[HOST],
    http_auth=(USER, PASS),
    use_ssl=True,
    verify_certs=True,
)

# -----------------------------------
# 2) CREATE INDEX (text-only, best-practice mapping)
# -----------------------------------
INDEX = "kb-text-index"

# Mapping notes:
# - title: text (analyzed) + keyword subfield for exact filtering/sorting
# - content: text (analyzed) - the main searchable body
# - tags/source: keyword (for filters)
# - created_at: date (ISO8601 strings)
settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "title":   {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "content": {"type": "text"},
            "tags":    {"type": "keyword"},
            "source":  {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
}

# Create if not exists
if not client.indices.exists(index=INDEX):
    client.indices.create(index=INDEX, body=settings)
    print(f"Index '{INDEX}' created.")
else:
    print(f"Index '{INDEX}' already exists.")

# -----------------------------------
# 3) BULK INSERT SAMPLE DOCS (mini KB)
# -----------------------------------
now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

docs = [
    {"title": "Intro to RAG",
     "content": "Retrieval Augmented Generation (RAG) combines an external knowledge search with an LLM to produce grounded answers.",
     "tags": ["rag", "overview"],
     "source": "lesson",
     "created_at": now},

    {"title": "Why Indexing Matters",
     "content": "Good indexing improves recall and precision. Clean fields, chunking, and metadata help retrieve the right passages quickly.",
     "tags": ["indexing", "best-practices"],
     "source": "lesson",
     "created_at": now},

    {"title": "Chunking Strategy",
     "content": "Split long documents into chunks of about 200 to 500 words so retrieval returns focused, relevant passages.",
     "tags": ["chunking", "best-practices"],
     "source": "lesson",
     "created_at": now},

    {"title": "OpenSearch Basics",
     "content": "OpenSearch supports full-text search with analyzers and keyword fields for exact matching and filters.",
     "tags": ["opensearch", "basics"],
     "source": "lesson",
     "created_at": now},

    {"title": "Fields and Mappings",
     "content": "Define mappings up front: use text for analyzed search, keyword for exact filtering, and date for timestamps.",
     "tags": ["mappings", "best-practices"],
     "source": "lesson",
     "created_at": now},

    {"title": "Query Types",
     "content": "Use match for full-text queries, multi-match to search multiple fields, and bool queries to combine conditions.",
     "tags": ["queries", "basics"],
     "source": "lesson",
     "created_at": now},

    {"title": "Phrase Matching",
     "content": "match_phrase helps when the word order is important, for example 'semantic search' as a phrase.",
     "tags": ["queries", "phrase"],
     "source": "lesson",
     "created_at": now},

    {"title": "Filters and Tags",
     "content": "Use keyword fields like tags to filter results without affecting relevance scoring.",
     "tags": ["filters", "tags"],
     "source": "lesson",
     "created_at": now},

    {"title": "Highlighting",
     "content": "Highlighting shows matched fragments, making search results easier to explain to students.",
     "tags": ["ux", "highlight"],
     "source": "lesson",
     "created_at": now},

    {"title": "Pagination",
     "content": "Use from and size to paginate. For large sets, prefer search_after with a stable sort.",
     "tags": ["pagination"],
     "source": "lesson",
     "created_at": now},

    {"title": "OpenSearch for RAG (Text Stage)",
     "content": "Start with clean text search and metadata filters before adding vector embeddings.",
     "tags": ["rag", "opensearch"],
     "source": "lesson",
     "created_at": now},

    {"title": "Metadata for RAG",
     "content": "Include fields like source, author, url, and created_at so answers can cite and filter context.",
     "tags": ["metadata", "rag"],
     "source": "lesson",
     "created_at": now},
]

actions = [
    {"_index": INDEX, "_source": d} for d in docs
]

helpers.bulk(client, actions)
# Ensure data is visible to searches
client.indices.refresh(index=INDEX)
print(f"Inserted {len(docs)} docs and refreshed index.\n")

# -----------------------------------
# Helpers to pretty print results
# -----------------------------------
def print_hits(res, show_highlight=True):
    hits = res.get("hits", {}).get("hits", [])
    print(f"Total hits (reported): {res.get('hits', {}).get('total', {})}\n")
    for i, h in enumerate(hits, 1):
        score = h.get("_score")
        src = h.get("_source", {})
        title = src.get("title")
        content = src.get("content", "")[:160].replace("\n", " ")
        print(f"{i}. _score={score:.3f} | {title}")
        print(f"   {content}...")
        if show_highlight and "highlight" in h:
            for field, frags in h["highlight"].items():
                for frag in frags:
                    print(f"   [HL] {field}: {frag}")
        print()

# -----------------------------------
# 4) BASIC QUERIES
# -----------------------------------
print("A) match (single field)\nQuery: 'search' in content\n")
q_match = {
    "query": {"match": {"content": "search"}},
    "highlight": {"fields": {"content": {}}}
}
res = client.search(index=INDEX, body=q_match, size=5)
print_hits(res)

print("B) multi_match (title^2 + content)\nQuery: 'OpenSearch search'\n")
q_multi = {
    "query": {
        "multi_match": {
            "query": "OpenSearch search",
            "fields": ["title^2", "content"]  # boost title
        }
    },
    "highlight": {"fields": {"title": {}, "content": {}}}
}
res = client.search(index=INDEX, body=q_multi, size=5)
print_hits(res)

print("C) phrase match\nQuery: 'semantic search' as a phrase\n")
q_phrase = {
    "query": {"match_phrase": {"content": "semantic search"}},
    "highlight": {"fields": {"content": {}}}
}
res = client.search(index=INDEX, body=q_phrase, size=5)
print_hits(res)

print("D) bool with filter on tags (keyword field)\nRequire content match:'RAG' AND filter tag:'rag'\n")
q_bool = {
    "query": {
        "bool": {
            "must": {"match": {"content": "RAG"}},
            "filter": {"terms": {"tags": ["rag"]}}
        }
    },
    "highlight": {"fields": {"content": {}}}
}
res = client.search(index=INDEX, body=q_bool, size=5)
print_hits(res)

print("E) Pagination example (from/size)\nFirst page: size=3\n")
q_page = {
    "query": {"match_all": {}},
    "sort": [{"_score": "desc"}]
}
res_page1 = client.search(index=INDEX, body=q_page, from_=0, size=3)
print_hits(res_page1, show_highlight=False)

print("Second page: from=3, size=3\n")
res_page2 = client.search(index=INDEX, body=q_page, from_=3, size=3)
print_hits(res_page2, show_highlight=False)

print("F) Terms aggregation on tags (what topics we have?)\n")
q_agg = {
    "size": 0,
    "aggs": {
        "tags_count": {
            "terms": {"field": "tags", "size": 20}
        }
    }
}
agg_res = client.search(index=INDEX, body=q_agg)
buckets = agg_res.get("aggregations", {}).get("tags_count", {}).get("buckets", [])
for b in buckets:
    print(f"tag='{b['key']}' -> {b['doc_count']} docs")

print("\nDone.")
