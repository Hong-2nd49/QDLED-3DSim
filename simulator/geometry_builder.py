from typing import Dict, List

def build_layered_geometry() -> List[Dict]:
    """
    Example: return a simple layered QLED-style stack definition.
    This is a logical geometry description, not a mesh.

    Each layer:
      - name: str
      - thickness_nm: float
      - material: str
    """
    return [
        {"name": "Anode", "material": "ITO", "thickness_nm": 150.0},
        {"name": "HTL", "material": "HTL", "thickness_nm": 40.0},
        {"name": "EML_QD", "material": "QD", "thickness_nm": 20.0},
        {"name": "ETL_ZnO", "material": "ZnO", "thickness_nm": 30.0},
        {"name": "Cathode", "material": "Al", "thickness_nm": 100.0},
    ]


def build_patterned_geometry(zno_ratio: float = 0.5) -> Dict:
    """
    Example of a simple lateral pattern descriptor.

    zno_ratio: fraction of lateral area covered by ZnO-contacted region.

    Returns:
      {
        "layers": [...],
        "lateral_pattern": {
            "ZnO_ratio": float,
            "description": str
        }
      }
    """
    geom = {
        "layers": build_layered_geometry(),
        "lateral_pattern": {
            "ZnO_ratio": float(zno_ratio),
            "description": "Simple stripe pattern with ZnO-contacted sub-regions."
        },
    }
    return geom
