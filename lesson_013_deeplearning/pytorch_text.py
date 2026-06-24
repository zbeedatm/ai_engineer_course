import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"  # sentiment demo
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label

@torch.no_grad()
def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    logits = model(**inputs).logits[0]
    probs = logits.softmax(dim=0)
    score, idx = probs.max(dim=0)
    return {"label": id2label[int(idx)], "score": float(score)}

if __name__ == "__main__":
    samples = [
        "This product is amazing and works perfectly.",
        "I hate this, it broke after one day."
    ]
    for s in samples:
        print(json.dumps({"text": s, "pred": predict_text(s)}, ensure_ascii=False))
