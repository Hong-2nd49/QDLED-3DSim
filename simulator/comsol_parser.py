import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

def parse_comsol_csv(path) -> Dict:
    """
    Parse COMSOL/TCAD-like CSV for QLED/QDLED simulations.

    Expected minimal columns:
      - x, z
      - (optional) y
      - n_electron
      - n_hole
      - R_rad      # radiative recombination rate density
      - R_nrad     # non-radiative recombination rate density

    Returns:
      {
        "df": DataFrame,
        "EQE_proxy": float,
        "overlap": float,
        "rad_fraction": float,
        "nonrad_fraction": float,
      }
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = {"x", "n_electron", "n_hole", "R_rad", "R_nrad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    ne = df["n_electron"].to_numpy()
    nh = df["n_hole"].to_numpy()
    R_rad = df["R_rad"].to_numpy()
    R_nrad = df["R_nrad"].to_numpy()

    # Electron-hole spatial overlap (normalized dot product)
    if np.all(ne == 0) or np.all(nh == 0):
        overlap = 0.0
    else:
        overlap = float(
            np.sum(ne * nh)
            / (np.sqrt(np.sum(ne**2)) * np.sqrt(np.sum(nh**2)) + 1e-12)
        )

    total_rad = float(np.sum(R_rad))
    total_nrad = float(np.sum(R_nrad))
    total = total_rad + total_nrad + 1e-18

    rad_fraction = total_rad / total
    nonrad_fraction = total_nrad / total

    # Internal EQE proxy
    EQE_proxy = rad_fraction

    return {
        "df": df,
        "EQE_proxy": EQE_proxy,
        "overlap": overlap,
        "rad_fraction": rad_fraction,
        "nonrad_fraction": nonrad_fraction,
    }
