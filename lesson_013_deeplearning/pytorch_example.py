import io
import json
from PIL import Image

import torch
import torchvision.models as models


# ----------- CONFIG -----------
IMAGE_PATH = r"dog-on-the-beach.jpg"   # <-- change this
TOPK = 5
FORCE_CPU = False  # set True if you want to force CPU even when GPU is available


def get_device():
    if FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device):
    # Load pretrained MobileNetV3 (ImageNet 1000 classes)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights).to(device).eval()

    preprocess = weights.transforms()              # correct preprocessing pipeline
    labels = weights.meta["categories"]            # list of 1000 label strings
    return model, preprocess, labels


@torch.no_grad()
def predict_image_path(model, preprocess, labels, image_path: str, device: torch.device, topk: int = 3):
    # Read image bytes from disk (like a pipeline would do)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Decode bytes -> PIL image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess -> tensor, add batch dim, move to device
    x = preprocess(img).unsqueeze(0).to(device)   # shape: (1, 3, H, W)

    # Run model -> logits -> probabilities
    logits = model(x)                              # shape: (1, 1000)
    probs = logits.softmax(dim=1)[0]               # shape: (1000,)

    # Top-K
    values, indices = probs.topk(topk)

    # JSON-friendly output
    result = [{"label": labels[int(i)], "prob": float(v)} for v, i in zip(values, indices)]
    return result


def main():
    device = get_device()
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    model, preprocess, labels = load_model(device)

    # Sanity check: verify model really on the device you think it is
    print("Model device:", next(model.parameters()).device)

    preds = predict_image_path(model, preprocess, labels, IMAGE_PATH, device, topk=TOPK)
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
