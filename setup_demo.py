"""One-command demo: generate synthetic data, build features, verify it loads.

Usage:
    python setup_demo.py

Works WITHOUT real MIMIC-IV data. Generates synthetic raw tables that match the
MIMIC-IV schema, builds the Stage 1 feature matrix, and prints a preview.
After this, run the modelling steps (see README / docs/MODELING_PLAN.md).
"""

from src.config import load_config, get_data_dir
from src.data.synthetic import generate_synthetic_dataset
from src.data.features import build_features, feature_columns
from src.schemas import TARGET_COL


def main():
    cfg = load_config()

    print("=" * 60)
    print("  Readmission Prediction — Demo Setup (Synthetic Data)")
    print("=" * 60, "\n")

    tables = generate_synthetic_dataset(
        n_patients=cfg.data.synthetic_n_patients,
        seed=cfg.data.synthetic_seed,
        output_dir=get_data_dir() / "synthetic",
    )

    print("\nBuilding feature matrix...")
    matrix = build_features(tables, cfg)
    out_dir = get_data_dir().parent / cfg.output.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_dir / "features.csv", index=False)

    print(f"  Feature matrix: {len(matrix):,} rows x {matrix.shape[1]} cols")
    print(f"  Model features: {len(feature_columns(matrix))}")
    print(f"  Readmission rate: {matrix[TARGET_COL].mean():.1%}")
    print(f"  Written to: {out_dir / 'features.csv'}")

    print("\nFeature preview:")
    print(matrix.head(5).to_string(index=False))
    print("\nDemo complete. Next: python -m src.model.train (see README).")


if __name__ == "__main__":
    main()
