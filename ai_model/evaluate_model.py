from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_model(artifacts_dir: str = "ai_model/artifacts"):
    meta_path = Path(artifacts_dir) / "meta.txt"
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    feature_cols = lines[0].split(",")
    target_cols = lines[1].split(",")

    model = SimpleMLP(in_dim=len(feature_cols), out_dim=len(target_cols))
    state = torch.load(Path(artifacts_dir) / "surrogate_mlp.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, feature_cols, target_cols


def predict_from_features(features: Dict[str, float], artifacts_dir: str = "ai_model/artifacts") -> Dict:
    model, feature_cols, target_cols = load_model(artifacts_dir)
    x = np.array([features[c] for c in feature_cols], dtype=float).reshape(1, -1)
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y = model(x_t).numpy().squeeze(0)
    return dict(zip(target_cols, y.tolist()))
