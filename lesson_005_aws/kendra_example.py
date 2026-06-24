import boto3

kendra = boto3.client("kendra")

response = kendra.query(
    IndexId="89dc508b-f32e-473a-b29d-f50f22288185",
    QueryText="What is the date of Risk Analysis Report?"
)

for item in response["ResultItems"]:
    print(item["DocumentExcerpt"]["Text"])