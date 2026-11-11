
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TEST_FILE = RAW_DIR / "test_FD001.txt"
RUL_FILE = RAW_DIR / "RUL_FD001.txt"

FEATURE_FILE = PROCESSED_DIR / "feature_columns.json"
METADATA_FILE = PROCESSED_DIR / "preprocessing_metadata.json"

MODEL_FILE = MODEL_DIR / "robust_random_forest.pkl"


# NASA C-MAPSS FD001 column names
COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    for path in [
        TEST_FILE,
        RUL_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    test_df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None
    )

    # NASA files may contain trailing empty columns.
    test_df = test_df.dropna(axis=1, how="all")

    if test_df.shape[1] != 26:
        raise ValueError(
            f"Expected 26 test columns, got {test_df.shape[1]}"
        )

    test_df.columns = COLUMNS

    true_rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    features = json.loads(
        FEATURE_FILE.read_text()
    )

    model = joblib.load(MODEL_FILE)

    return test_df, true_rul, features, model


# --------------------------------------------------
# 3. EXACT FEATURE ENGINEERING
# --------------------------------------------------

def create_features(data, window=5):

    df = data.copy()

    df = df.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    sensor_cols = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    for sensor in sensor_cols:

        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{sensor}_rolling_std"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
            .fillna(0)
        )

        df[f"{sensor}_change"] = (
            df.groupby("unit_id")[sensor]
            .diff()
            .fillna(0)
        )

    return df


# --------------------------------------------------
# 4. PREPARE OFFICIAL TEST LABELS
# --------------------------------------------------

def prepare_test_labels(test_df, true_rul):

    last_cycles = (
        test_df.groupby("unit_id")["cycle"]
        .max()
        .sort_index()
    )

    engine_ids = last_cycles.index.to_numpy()

    if len(engine_ids) != len(true_rul):
        raise ValueError(
            "Test engine count does not match NASA RUL label count."
        )

    labels = pd.DataFrame({
        "unit_id": engine_ids,
        "last_cycle": last_cycles.values,
        "true_rul": true_rul.to_numpy()
    })

    return labels


# --------------------------------------------------
# 5. GENERATE FINAL-CYCLE FEATURES
# --------------------------------------------------

def prepare_test_features(test_df, features):

    engineered = create_features(
        test_df,
        window=5
    )

    # Keep the final observed row for each engine.
    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    missing_features = [
        feature for feature in features
        if feature not in final_rows.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X_test = final_rows[features].copy()

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    return final_rows, X_test


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

def evaluate():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading NASA FD001 test data...")

    test_df, true_rul, features, model = load_data()

    print("Test data shape:", test_df.shape)
    print("Number of engines:", test_df["unit_id"].nunique())
    print("Model feature count:", len(features))

    labels = prepare_test_labels(
        test_df,
        true_rul
    )

    final_rows, X_test = prepare_test_features(
        test_df,
        features
    )

    if not np.array_equal(
        final_rows["unit_id"].to_numpy(),
        labels["unit_id"].to_numpy()
    ):
        raise ValueError("Engine ordering mismatch.")

    predictions = model.predict(X_test)

    results = labels.copy()

    results["predicted_rul"] = predictions

    results["absolute_error"] = np.abs(
        results["true_rul"] - results["predicted_rul"]
    )

    results["error"] = (
        results["predicted_rul"] - results["true_rul"]
    )

    mae = mean_absolute_error(
        results["true_rul"],
        results["predicted_rul"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results["true_rul"],
            results["predicted_rul"]
        )
    )

    metrics = {
        "dataset": "NASA C-MAPSS FD001 official test set",
        "model": "Robust Random Forest",
        "engine_count": int(len(results)),
        "feature_count": int(len(features)),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }

    results.to_csv(
        REPORT_DIR / "nasa_test_predictions.csv",
        index=False
    )

    with open(
        REPORT_DIR / "nasa_test_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\n----------- OFFICIAL TEST RESULTS -----------")
    print("Engine count:", len(results))
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\nFirst five engine predictions:")
    print(
        results.head(5).to_string(index=False)
    )

    print("\nSaved:")
    print("reports/nasa_test_predictions.csv")
    print("reports/nasa_test_metrics.json")


if __name__ == "__main__":
    evaluate()
