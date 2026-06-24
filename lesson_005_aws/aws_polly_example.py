import boto3

polly = boto3.client("polly")

response = polly.synthesize_speech(
    Text="Hello! Welcome to our AI course.",
    OutputFormat="mp3",
    VoiceId="Joanna"
)

with open("welcome.mp3", "wb") as f:
    f.write(response["AudioStream"].read())
