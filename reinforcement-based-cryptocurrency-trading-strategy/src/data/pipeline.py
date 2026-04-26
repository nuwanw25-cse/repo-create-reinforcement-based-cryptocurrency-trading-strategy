"""
Full data pipeline: load → clean → features → split → normalize → save.

Run from the project root with:
    python -m src.data.pipeline

Reads config from configs/experiment_config.yaml.
Outputs:
    data/processed/   — cleaned OHLCV CSVs for train / val / test
    data/features/    — feature-enriched CSVs for train / val / test (un-normalised)
    data/normalized/  — fully normalised feature CSVs ready for the RL environment
"""
import yaml
import pandas as pd
from pathlib import Path

from src.data.loader import load_multiple, validate_continuity
from src.data.preprocessor import clean, normalize
from src.data.features import build_features, drop_warmup

CONFIG_PATH = Path("configs/experiment_config.yaml")


def run(config_path: Path = CONFIG_PATH) -> dict:
    """
    Execute the full data pipeline.

    Returns a dict with keys:
        train, val, test          — cleaned OHLCV DataFrames
        train_feat, val_feat, test_feat — feature-enriched DataFrames
        train_norm, val_norm, test_norm — normalised feature DataFrames
        scalers                   — {"price": scaler, "volume": scaler}
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feat_cfg = {
        "sma_short":  data_cfg["sma_short"],
        "sma_long":   data_cfg["sma_long"],
        "rsi_period": data_cfg["rsi_period"],
    }

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("STEP 1 — Load raw CSV files")
    print("=" * 60)
    train_raw = load_multiple(data_cfg["train_files"])
    val_raw   = load_multiple(data_cfg["val_files"])
    test_raw  = load_multiple(data_cfg["test_files"])

    validate_continuity(train_raw)
    validate_continuity(val_raw)
    validate_continuity(test_raw)

    # ------------------------------------------------------------------ #
    # 2. Clean
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 2 — Clean")
    print("=" * 60)
    print("train: ", end=""); train_raw = clean(train_raw)
    print("val:   ", end=""); val_raw   = clean(val_raw)
    print("test:  ", end=""); test_raw  = clean(test_raw)

    # Save processed (clean OHLCV, no indicators yet)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_raw.to_csv(processed_dir / "train.csv")
    val_raw.to_csv(processed_dir / "val.csv")
    test_raw.to_csv(processed_dir / "test.csv")
    print(f"\nSaved cleaned CSVs → {processed_dir}/")

    # ------------------------------------------------------------------ #
    # 3. Feature engineering
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 3 — Feature engineering")
    print("=" * 60)

    # Compute on full combined dataset so SMA-50 warm-up rows stay in train
    combined = pd.concat([train_raw, val_raw, test_raw]).sort_index()
    combined = build_features(combined, feat_cfg)

    # Split back by date
    train_feat = combined.loc[train_raw.index]
    val_feat   = combined.loc[val_raw.index]
    test_feat  = combined.loc[test_raw.index]

    # Drop warm-up NaN rows from training only
    train_feat = drop_warmup(train_feat)

    print(f"\nTrain : {len(train_feat)} rows  "
          f"({train_feat.index[0].date()} → {train_feat.index[-1].date()})")
    print(f"Val   : {len(val_feat)} rows  "
          f"({val_feat.index[0].date()} → {val_feat.index[-1].date()})")
    print(f"Test  : {len(test_feat)} rows  "
          f"({test_feat.index[0].date()} → {test_feat.index[-1].date()})")

    # Save feature CSVs (un-normalised — useful for EDA notebooks)
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(features_dir / "train.csv")
    val_feat.to_csv(features_dir / "val.csv")
    test_feat.to_csv(features_dir / "test.csv")
    print(f"\nSaved feature CSVs → {features_dir}/")

    # ------------------------------------------------------------------ #
    # 4. Normalize (fit on train only)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 4 — Normalize")
    print("=" * 60)
    train_norm, val_norm, test_norm, scalers = normalize(
        train_feat, val_feat, test_feat
    )

    # Save fully-normalised feature CSVs — these are the input for TradingEnv
    normalized_dir = Path("data/normalized")
    normalized_dir.mkdir(parents=True, exist_ok=True)
    train_norm.to_csv(normalized_dir / "train.csv")
    val_norm.to_csv(normalized_dir / "val.csv")
    test_norm.to_csv(normalized_dir / "test.csv")
    print(f"Saved normalized feature CSVs → {normalized_dir}/")

    print("\nPipeline complete.")
    print(f"  Train : {len(train_norm)} rows")
    print(f"  Val   : {len(val_norm)} rows")
    print(f"  Test  : {len(test_norm)} rows")

    return {
        "train": train_raw, "val": val_raw, "test": test_raw,
        "train_feat": train_feat, "val_feat": val_feat, "test_feat": test_feat,
        "train_norm": train_norm, "val_norm": val_norm, "test_norm": test_norm,
        "scalers": scalers,
    }


if __name__ == "__main__":
    run()
