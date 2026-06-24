import boto3

translate = boto3.client("translate", region_name="us-east-1")

result = translate.translate_text(
    Text="Welcome to the AI engineering Course!",
    SourceLanguageCode="en",
    TargetLanguageCode="he"
)

print(result["TranslatedText"])
