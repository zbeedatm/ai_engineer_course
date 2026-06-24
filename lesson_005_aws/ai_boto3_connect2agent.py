import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

agent_id = "TJHSCJRGOK"
agent_alias_id = "KHWS6C7BZ9"
session_id = "example-session-001"

response = client.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=session_id,
    inputText="What is the current time in TLV?" #"What is the capital of France?"
)

for event in response.get("completion", []):
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode("utf-8"))
    elif "trace" in event:
        print("Trace:", event["trace"])