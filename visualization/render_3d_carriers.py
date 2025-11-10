import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_carrier_slice(csv_path: str, z_level: float):
    """
    Plot n_electron and n_hole vs x at a given z slice (approximate).
    """
    df = pd.read_csv(Path(csv_path))

    if "z" not in df.columns:
        raise ValueError("CSV must contain 'z' for slice plotting.")
    if "x" not in df.columns or "n_electron" not in df.columns or "n_hole" not in df.columns:
        raise ValueError("CSV missing required columns.")

    # Simple nearest slice
    idx = (df["z"] - z_level).abs().idxmin()
    z_sel = df.loc[idx, "z"]
    slice_df = df[df["z"] == z_sel]

    plt.figure()
    plt.plot(slice_df["x"], slice_df["n_electron"], label="n_electron")
    plt.plot(slice_df["x"], slice_df["n_hole"], label="n_hole")
    plt.xlabel("x")
    plt.ylabel("Carrier density")
    plt.title(f"Carrier densities at z = {z_sel:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()
