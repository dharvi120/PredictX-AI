
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "unit_id",
        "cycle",
        "max_cycle",
        "RUL"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["RUL"].isna().any():
        raise ValueError("RUL contains missing values.")

    if (df["RUL"] < 0).any():
        raise ValueError("Negative RUL values found.")

    print("Dataset loaded successfully.")
    print("Original shape:", df.shape)

    return df


# --------------------------------------------------
# 3. SPLIT BY ENGINE ID
# --------------------------------------------------

def split_by_engine(df):

    engine_ids = df["unit_id"].unique()

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=0.20,
        random_state=42
    )

    train_df = df[df["unit_id"].isin(train_ids)].copy()
    validation_df = df[
        df["unit_id"].isin(validation_ids)
    ].copy()

    if set(train_ids).intersection(set(validation_ids)):
        raise ValueError("Engine leakage detected!")

    print("\nEngine-wise split completed.")
    print("Training engines:", len(train_ids))
    print("Validation engines:", len(validation_ids))

    return train_df, validation_df


# --------------------------------------------------
# 4. SELECT MODEL FEATURES
# --------------------------------------------------

def select_features(train_df, validation_df):

    excluded_columns = [
        "unit_id",
        "max_cycle",
        "RUL"
    ]

    feature_columns = [
        col for col in train_df.columns
        if col not in excluded_columns
    ]

    non_numeric = [
        col for col in feature_columns
        if not pd.api.types.is_numeric_dtype(
            train_df[col]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric features found: {non_numeric}"
        )

    # Remove constant features using training data only.
    constant_columns = [
        col for col in feature_columns
        if train_df[col].nunique(dropna=False) <= 1
    ]

    feature_columns = [
        col for col in feature_columns
        if col not in constant_columns
    ]

    if not feature_columns:
        raise ValueError("No usable features remain.")

    for name, dataset in [
        ("Training", train_df),
        ("Validation", validation_df)
    ]:
        values = dataset[feature_columns].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} features contain NaN or infinity."
            )

    print("\nFeature selection completed.")
    print("Removed constant features:", constant_columns)
    print("Final feature count:", len(feature_columns))

    return feature_columns


# --------------------------------------------------
# 5. SAVE PREPROCESSED DATA
# --------------------------------------------------

def save_outputs(
    train_df,
    validation_df,
    feature_columns
):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "unit_id",
        *feature_columns,
        "RUL"
    ]

    train_final = train_df[output_columns].copy()
    validation_final = validation_df[output_columns].copy()

    train_final.to_csv(
        OUTPUT_DIR / "model_train.csv",
        index=False
    )

    validation_final.to_csv(
        OUTPUT_DIR / "model_validation.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "feature_columns.json",
        "w"
    ) as file:
        json.dump(feature_columns, file, indent=4)

    metadata = {
        "input_file": str(INPUT_FILE),
        "training_rows": len(train_final),
        "validation_rows": len(validation_final),
        "training_engines": int(
            train_final["unit_id"].nunique()
        ),
        "validation_engines": int(
            validation_final["unit_id"].nunique()
        ),
        "feature_count": len(feature_columns),
        "target": "RUL",
        "split_method": "Engine-wise",
        "test_size": 0.20,
        "random_state": 42
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nAll files saved successfully.")
    print("Training shape:", train_final.shape)
    print("Validation shape:", validation_final.shape)
    print("Output directory:", OUTPUT_DIR)

    return train_final, validation_final


# --------------------------------------------------
# 6. MAIN PIPELINE
# --------------------------------------------------

def prepare_datasets():

    df = load_data()

    train_df, validation_df = split_by_engine(df)

    feature_columns = select_features(
        train_df,
        validation_df
    )

    return save_outputs(
        train_df,
        validation_df,
        feature_columns
    )


if __name__ == "__main__":

    train_final, validation_final = prepare_datasets()

    print("\nDAY 6 PREPROCESSING COMPLETED!")
