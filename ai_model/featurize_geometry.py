from typing import Dict, List
import numpy as np

def featurize_geometry(geometry: Dict) -> np.ndarray:
    """
    Convert a simple geometry dict into a numeric feature vector.

    Example features:
      - number of layers
      - total thickness
      - QD layer thickness
      - ZnO ETL thickness
      - ZnO lateral ratio (if present)
    """
    layers: List[Dict] = geometry.get("layers", [])
    n_layers = len(layers)
    total_thickness = sum(l["thickness_nm"] for l in layers)

    qd_th = sum(l["thickness_nm"] for l in layers if "QD" in l["material"])
    zno_th = sum(l["thickness_nm"] for l in layers if "ZnO" in l["material"])

    lateral = geometry.get("lateral_pattern", {})
    zno_ratio = float(lateral.get("ZnO_ratio", 0.0))

    return np.array(
        [n_layers, total_thickness, qd_th, zno_th, zno_ratio],
        dtype=float,
    )
