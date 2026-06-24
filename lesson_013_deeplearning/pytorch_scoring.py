import json
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretend this is trained already and loaded from a checkpoint
class RiskMLP(nn.Module):
    def __init__(self, n_features=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # score logit
        )
    def forward(self, x):
        return self.net(x)

model = RiskMLP().to(device).eval()

@torch.no_grad()
def score(features: list[float]) -> float:
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)  # (1, n)
    logit = model(x)[0, 0]
    prob = torch.sigmoid(logit)  # 0..1 score
    return float(prob)

if __name__ == "__main__":
    event = {"user_id": 7, "features": [0.1, 12.0, 0.0, 3.4, 8.8, 1.0]}
    out = {"user_id": event["user_id"], "risk_score": score(event["features"])}
    print(json.dumps(out))
