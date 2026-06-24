import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

host = os.environ.get("OPENSEARCH_HOST")
user = os.environ.get("OPENSEARCH_USER", "admin")
password = os.environ.get("OPENSEARCH_PASS")

client = OpenSearch(
    hosts=[host],
    http_auth=(user, password),
    use_ssl=True,
    verify_certs=True
)