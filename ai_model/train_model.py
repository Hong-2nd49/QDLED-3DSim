import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .featurize_geometry import featurize_geometry

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


def main():
    parser = argparse.ArgumentParser(description="Train surrogate model on preprocessed data.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/preprocessed/training_data.csv",
        help="CSV with geometry features and targets.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No training CSV found at {csv_path}")

    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    target_cols = ["EQE_proxy", "overlap"]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_cols].to_numpy(dtype=float)

    model = SimpleMLP(in_dim=X.shape[1], out_dim=y.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(300):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"[Epoch {epoch+1}] loss={loss.item():.6f}")

    out_dir = Path("ai_model/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "surrogate_mlp.pt")

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(",".join(feature_cols) + "\n")
        f.write(",".join(target_cols) + "\n")

    print(f"Saved model and meta to {out_dir}")


if __name__ == "__main__":
    main()
