# claude_basic.py
import json
import boto3
from botocore.exceptions import ClientError

REGION   = "us-east-1"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # <-- put your exact Sonnet model ID here
# MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"

def claude_complete(prompt: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.2,
        # optional "system": "You are a concise assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }

    try:
        resp = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        payload = resp["body"].read() if hasattr(resp.get("body"), "read") else resp["body"]
        data = json.loads(payload)

        # Anthropic messages return a list of content blocks; join any text blocks.
        parts = data.get("content", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return text.strip()

    except ClientError as e:
        raise RuntimeError(f"Bedrock InvokeModel failed: {e.response.get('Error', {}).get('Message')}") from e

if __name__ == "__main__":
    out = claude_complete("Give me three bullet points about why RAG is useful.")
    print(out)
