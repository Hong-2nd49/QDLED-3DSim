import argparse
from pathlib import Path

from simulator.comsol_parser import parse_comsol_csv
from simulator.geometry_builder import build_layered_geometry
from ai_model.featurize_geometry import featurize_geometry

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Parse simulation file and compute basic metrics.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_simulations/example_simulation.csv",
        help="Path to COMSOL/TCAD-like CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Parse simulation data
    result = parse_comsol_csv(input_path)
    df = result["df"]
    EQE_proxy = result["EQE_proxy"]
    overlap = result["overlap"]
    rad_frac = result["rad_fraction"]
    nonrad_frac = result["nonrad_fraction"]

    print(f"Parsed {len(df)} points from {input_path}")
    print(f"Internal EQE proxy (radiative fraction): {EQE_proxy:.3f}")
    print(f"e–h overlap: {overlap:.3f}")
    print(f"Radiative fraction: {rad_frac:.3f}, Non-radiative fraction: {nonrad_frac:.3f}")

    # Example: attach simple geometry features and save preprocessed row
    geom = build_layered_geometry()
    feat_vec = featurize_geometry({"layers": geom})

    out_row = {
        "feat_n_layers": feat_vec[0],
        "feat_total_thickness": feat_vec[1],
        "feat_qd_thickness": feat_vec[2],
        "feat_zno_thickness": feat_vec[3],
        "feat_zno_ratio": feat_vec[4],
        "EQE_proxy": EQE_proxy,
        "overlap": overlap,
    }

    out_dir = Path("data/preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "example_processed.csv"
    pd.DataFrame([out_row]).to_csv(out_csv, index=False)

    print(f"Saved processed metrics to {out_csv}")

if __name__ == "__main__":
    main()
