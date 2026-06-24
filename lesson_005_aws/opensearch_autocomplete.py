"""
OpenSearch Indexing Patterns — Practical Examples
Requires: pip install opensearch-py

This script demonstrates:
1) Autocomplete with edge-n-grams
2) Completion suggester
3) Synonyms + normalizer
4) Nested objects + inner_hits
5) Vector index with HNSW (demo vectors)
6) Hybrid search (BM25 + vectors) with RRF search pipeline (if supported)
7) Ingest pipeline (transform on write)
8) Index template (reuse settings/mappings)

NOTE: Replace HOST/USER/PASS to match your domain.
"""

import os
import random
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

load_dotenv()

# -----------------------------
# Connection
# -----------------------------
HOST = os.environ.get("OPENSEARCH_HOST")
USER = os.environ.get("OPENSEARCH_USER", "admin")
PASS = os.environ.get("OPENSEARCH_PASS")

client = OpenSearch(
    hosts=[HOST],
    http_auth=(USER, PASS),
    use_ssl=True,
    verify_certs=True,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

def ptitle(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def print_hits(res, show_highlight=True, max_chars=160):
    hits = res.get("hits", {}).get("hits", [])
    total = res.get("hits", {}).get("total", {})
    print(f"Total: {total}\n")
    for i, h in enumerate(hits, 1):
        _id = h.get("_id")
        score = h.get("_score")
        src = h.get("_source", {})
        title = src.get("title", "")
        content = str(src.get("content", ""))[:max_chars].replace("\n", " ")
        print(f"{i}. _id={_id}  _score={score}  title={title}")
        if content:
            print(f"   {content}...")
        if show_highlight and "highlight" in h:
            for field, frags in h["highlight"].items():
                for frag in frags:
                    print(f"   [HL] {field}: {frag}")
        print()

# --------------------------------------------------------------------
# 1) Autocomplete with edge-n-grams (type-ahead)
# --------------------------------------------------------------------
def example_autocomplete_edge_ngram():
    index = "kb-autocomplete"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "settings": {
            "analysis": {
                "filter": {
                    "edge_ngram_filter": {
                        "type": "edge_ngram",
                        "min_gram": 1,
                        "max_gram": 20
                    }
                },
                "analyzer": {
                    "autocomplete": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "edge_ngram_filter"]
                    },
                    "autocomplete_search": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "autocomplete",
                    "search_analyzer": "autocomplete_search"
                }
            }
        }
    }
    client.indices.create(index=index, body=body)

    docs = [
        {"_index": index, "_source": {"title": "OpenSearch Basics"}},
        {"_index": index, "_source": {"title": "OpenSearch Vector Search"}},
        {"_index": index, "_source": {"title": "Operational Excellence"}},
        {"_index": index, "_source": {"title": "OpenAI Integration"}},
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=index)

    ptitle("1) Autocomplete (edge-n-gram) — query: 'open se'")
    q = {"query": {"match": {"title": "open se"}}, "size": 5}
    res = client.search(index=index, body=q)
    print_hits(res, show_highlight=False)

# --------------------------------------------------------------------
# 2) Completion suggester
# --------------------------------------------------------------------
def example_completion_suggester():
    index = "kb-suggest"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "mappings": {
            "properties": {
                "suggest": {"type": "completion"},
                "title":   {"type": "text"}
            }
        }
    }
    client.indices.create(index=index, body=body)

    to_index = [
        {"title": "OpenSearch Basics", "suggest": ["OpenSearch Basics"]},
        {"title": "OpenSearch Dashboards", "suggest": ["OpenSearch Dashboards"]},
        {"title": "Operational Excellence", "suggest": ["Operational Excellence"]},
        {"title": "OpenAI Integration", "suggest": ["OpenAI Integration"]},
    ]
    for d in to_index:
        client.index(index=index, body=d)
    client.indices.refresh(index=index)

    ptitle("2) Completion suggester — prefix: 'open'")
    q = {
        "suggest": {
            "kb": {
                "prefix": "open",
                "completion": {"field": "suggest", "fuzzy": {"fuzziness": 1}}
            }
        }
    }
    res = client.search(index=index, body=q)
    print(res.get("suggest", {}), "\n")

# --------------------------------------------------------------------
# 3) Synonyms + normalizer for keyword filters
# --------------------------------------------------------------------
def example_synonyms_normalizer():
    index = "kb-synonyms"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "settings": {
            "analysis": {
                "filter": {
                    "my_syns": {
                        "type": "synonym_graph",
                        "synonyms": [
                            "electric car, EV",
                            "sports car, race car"
                        ]
                    }
                },
                "analyzer": {
                    "syns_search": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "my_syns"]
                    }
                },
                "normalizer": {
                    "norm_kw": {
                        "type": "custom",
                        "filter": ["asciifolding", "lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text", "search_analyzer": "syns_search"},
                "tag":     {"type": "keyword", "normalizer": "norm_kw"}
            }
        }
    }
    client.indices.create(index=index, body=body)

    docs = [
        {"_index": index, "_source": {"content": "EV charging networks are expanding fast.", "tag": "RAG"}},
        {"_index": index, "_source": {"content": "Electric car adoption depends on charging.", "tag": "rag"}},
        {"_index": index, "_source": {"content": "Race car aerodynamics are different.", "tag": "Auto"}},
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=index)

    ptitle("3) Synonyms + normalizer — query: 'EV charging' + filter tag:'Rag'")
    q = {
        "query": {
            "bool": {
                "must": {"match": {"content": "EV charging"}},
                "filter": {"term": {"tag": "Rag"}}  # case-insensitive match due to normalizer
            }
        }
    }
    res = client.search(index=index, body=q)
    print_hits(res, show_highlight=False)

# --------------------------------------------------------------------
# 4) Nested objects + inner_hits
# --------------------------------------------------------------------
def example_nested_inner_hits():
    index = "kb-nested"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "mappings": {
            "properties": {
                "doc_title": {"type": "text"},
                "sections": {
                    "type": "nested",
                    "properties": {
                        "heading": {"type": "text"},
                        "text":    {"type": "text"}
                    }
                }
            }
        }
    }
    client.indices.create(index=index, body=body)

    d = {
        "doc_title": "Vector Indexing Guide",
        "sections": [
            {"heading": "Intro", "text": "Vectors enable semantic search."},
            {"heading": "Latency", "text": "Tuning HNSW parameters improves vector search latency."},
            {"heading": "Quality", "text": "Chunking influences retrieval quality."}
        ]
    }
    client.index(index=index, body=d)
    client.indices.refresh(index=index)

    ptitle("4) Nested query with inner_hits — match 'vector search latency' in sections.text")
    q = {
        "query": {
            "nested": {
                "path": "sections",
                "query": {"match": {"sections.text": "vector search latency"}},
                "inner_hits": {"size": 3}
            }
        }
    }
    res = client.search(index=index, body=q)
    # print main hit
    print_hits(res, show_highlight=False)
    # print inner hits
    hits = res.get("hits", {}).get("hits", [])
    if hits and "inner_hits" in hits[0]:
        print("Inner hits:")
        for ih in hits[0]["inner_hits"]["sections"]["hits"]["hits"]:
            print(" -", ih["_source"])

# --------------------------------------------------------------------
# 5) Vector index with HNSW (demo vectors)
# --------------------------------------------------------------------
def randvec(d=8):
    return [random.random() for _ in range(d)]

def example_vectors_hnsw():
    index = "kb-vectors"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 8,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"m": 16, "ef_construction": 128}
                    }
                }
            }
        }
    }
    client.indices.create(index=index, body=body)

    docs = [
        {"_index": index, "_source": {"content": "RAG connects AI with external knowledge.", "embedding": randvec(8)}},
        {"_index": index, "_source": {"content": "Vector databases store embeddings for semantic search.", "embedding": randvec(8)}},
        {"_index": index, "_source": {"content": "BM25 is lexical; vectors are semantic.", "embedding": randvec(8)}},
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=index)

    ptitle("5) Vector KNN — k=2 on a random query vector (demo)")
    qvec = randvec(8)  # In real RAG: use Titan/OpenAI embedding for the query text
    q = {
        "size": 2,
        "query": {
            "knn": {
                "embedding": {
                    "vector": qvec,
                    "k": 2
                }
            }
        }
    }
    res = client.search(index=index, body=q)
    print_hits(res, show_highlight=False)

# --------------------------------------------------------------------
# 6) Hybrid search (BM25 + vectors) with RRF search pipeline (if supported)
# --------------------------------------------------------------------
def example_hybrid_rrf():
    index = "kb-hybrid"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "title":   {"type": "text"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 8,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"m": 16, "ef_construction": 128}
                    }
                }
            }
        }
    }
    client.indices.create(index=index, body=body)

    docs = [
        {"_index": index, "_source": {"title": "Index vectors", "content": "How to index vectors in OpenSearch", "embedding": randvec(8)}},
        {"_index": index, "_source": {"title": "Hybrid search", "content": "Combine BM25 with vector search using RRF", "embedding": randvec(8)}},
        {"_index": index, "_source": {"title": "BM25 basics",   "content": "Lexical search in OpenSearch uses BM25", "embedding": randvec(8)}},
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=index)

    # Create an RRF search pipeline (if your cluster supports search pipelines)
    try:
        client.transport.perform_request(
            "PUT",
            "/_search/pipeline/rrf-pipeline",
            body={
                "description": "RRF for hybrid BM25+vector",
                "phase_results_processors": [
                    {
                        "score-ranker-processor": {
                            "combination": {
                                "technique": "rrf",
                                "rank_constant": 60
                            }
                        }
                    }
                ]
            }
        )
        created_pipeline = True
    except Exception as e:
        created_pipeline = False
        print("Could not create RRF search pipeline (cluster may be < 2.19 or missing feature):", e)

    ptitle("6) Hybrid search (BM25 + vector) with RRF (if available)")
    query_text = "how to index vectors"
    qvec = randvec(8)  # In real RAG: use actual embedding of query_text

    # Newer OpenSearch supports a 'hybrid' query array. Fall back if unsupported.
    hybrid_query = {
        "query": {
            "hybrid": [
                {"knn": {"embedding": {"vector": qvec, "k": 50}}},
                {"multi_match": {"query": query_text, "fields": ["title^2", "content"]}}
            ]
        },
        "size": 5
    }

    try:
        if created_pipeline:
            res = client.search(index=index, body=hybrid_query, params={"search_pipeline": "rrf-pipeline"})
        else:
            # If pipeline unsupported, run without it; some clusters still accept 'hybrid'
            res = client.search(index=index, body=hybrid_query)
        print_hits(res, show_highlight=False)
    except Exception as e:
        print("Hybrid query not supported on this cluster/version:", e)
        # Fallback: run separate queries and print both
        print("\nFallback: BM25 only")
        res_bm25 = client.search(index=index, body={
            "query": {"multi_match": {"query": query_text, "fields": ["title^2","content"]}}, "size": 5
        })
        print_hits(res_bm25, show_highlight=False)

        print("Fallback: KNN only")
        res_knn = client.search(index=index, body={
            "query": {"knn": {"embedding": {"vector": qvec, "k": 5}}}, "size": 5
        })
        print_hits(res_knn, show_highlight=False)

# --------------------------------------------------------------------
# 7) Ingest pipeline (transform on write)
# --------------------------------------------------------------------
def example_ingest_pipeline():
    # Create a tiny target index
    index = "kb-ingest-target"
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[400, 404])

    client.indices.create(index=index, body={
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "ingested_at": {"type": "date"}
            }
        }
    })

    # Create ingest pipeline: set timestamp, remove a debug field
    try:
        client.transport.perform_request(
            "PUT",
            "/_ingest/pipeline/kb-clean",
            body={
                "processors": [
                    {"set": {"field": "ingested_at", "value": "{{_ingest.timestamp}}"}},
                    {"remove": {"field": "debug"}}
                ]
            }
        )
    except Exception as e:
        print("Failed to create ingest pipeline:", e)

    ptitle("7) Ingest pipeline — set timestamp, remove 'debug'")
    # Index a doc through the pipeline
    client.index(index=index, body={
        "title": "RAG Intro",
        "content": "Transform fields during ingest with pipelines.",
        "debug": "tmp"
    }, params={"pipeline": "kb-clean"})
    client.indices.refresh(index=index)

    res = client.search(index=index, body={"query": {"match_all": {}}})
    print_hits(res, show_highlight=False)

# --------------------------------------------------------------------
# 8) Index template (reuse settings/mappings across kb-* indexes)
# --------------------------------------------------------------------
def example_index_template():
    template = {
        "index_patterns": ["kb-*"],
        "template": {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "title":   {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                    "content": {"type": "text"},
                    "tags":    {"type": "keyword"}
                }
            }
        },
        "priority": 100
    }
    try:
        client.transport.perform_request("PUT", "/_index_template/kb-template", body=template)
        created = True
    except Exception as e:
        created = False
        print("Failed to put index template (requires API support):", e)

    ptitle("8) Index template — create kb-2025-09-04 to auto-apply template")
    if created:
        idx = "kb-2025-09-04"
        if client.indices.exists(index=idx):
            client.indices.delete(index=idx, ignore=[400, 404])
        client.indices.create(index=idx)  # Template applies automatically
        # Verify by indexing & searching
        client.index(index=idx, body={"title": "Template Doc", "content": "This used the kb-template.", "tags": ["demo"]})
        client.indices.refresh(index=idx)
        res = client.search(index=idx, body={"query": {"match_all": {}}})
        print_hits(res, show_highlight=False)
    else:
        print("Template creation not supported on this cluster; skipping index creation test.")

# -----------------------------
# Run all examples
# -----------------------------
if __name__ == "__main__":
    example_autocomplete_edge_ngram()
    example_completion_suggester()
    example_synonyms_normalizer()
    example_nested_inner_hits()
    example_vectors_hnsw()
    example_hybrid_rrf()
    example_ingest_pipeline()
    example_index_template()
    print("\nAll examples finished.")
