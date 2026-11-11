
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

VALIDATION_FILE = DATA_DIR / "model_validation.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.json"

MODEL_FILE = MODEL_DIR / "robust_random_forest.pkl"

ALPHA = 0.10
RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

def load_inputs():

    for path in [
        VALIDATION_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    df = pd.read_csv(VALIDATION_FILE)

    with open(FEATURE_FILE, "r") as file:
        features = json.load(file)

    model = joblib.load(MODEL_FILE)

    required = ["unit_id", "RUL"] + features

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values found in validation dataset."
        )

    if not np.isfinite(
        df[features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in validation features."
        )

    return df, features, model


# --------------------------------------------------
# 3. SPLIT VALIDATION ENGINES
# --------------------------------------------------

def split_engines(df):

    engine_ids = np.sort(df["unit_id"].unique())

    if len(engine_ids) < 4:
        raise ValueError(
            "Too few engines for calibration and testing."
        )

    calibration_ids, test_ids = train_test_split(
        engine_ids,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    calibration_df = df[
        df["unit_id"].isin(calibration_ids)
    ].copy()

    test_df = df[
        df["unit_id"].isin(test_ids)
    ].copy()

    overlap = set(calibration_ids).intersection(
        set(test_ids)
    )

    if overlap:
        raise ValueError(
            "Calibration and test engine leakage detected."
        )

    print("\nEngine split completed.")
    print("Total validation engines:", len(engine_ids))
    print("Calibration engines:", len(calibration_ids))
    print("Test engines:", len(test_ids))

    return calibration_df, test_df


# --------------------------------------------------
# 4. CALIBRATION SCORES
# --------------------------------------------------

def calculate_calibration_scores(
    model,
    calibration_df,
    features
):

    X_cal = calibration_df[features]
    y_cal = calibration_df["RUL"].to_numpy()

    predictions = model.predict(X_cal)

    calibration_data = calibration_df[
        ["unit_id", "cycle"]
    ].copy()

    calibration_data["absolute_error"] = np.abs(
        y_cal - predictions
    )

    # One maximum error per engine.
    engine_scores = (
        calibration_data.groupby("unit_id")[
            "absolute_error"
        ]
        .max()
    )

    return engine_scores


# --------------------------------------------------
# 5. CONFORMAL QUANTILE
# --------------------------------------------------

def calculate_quantile(scores, alpha):

    scores = np.asarray(scores, dtype=float)

    n = len(scores)

    if n == 0:
        raise ValueError(
            "No calibration scores available."
        )

    # Finite-sample split conformal quantile.
    rank = math.ceil((n + 1) * (1 - alpha))

    # If rank exceeds n, use infinity for the
    # conservative finite-sample convention.
    if rank > n:
        return float("inf")

    return float(
        np.sort(scores)[rank - 1]
    )


# --------------------------------------------------
# 6. GENERATE PREDICTION INTERVALS
# --------------------------------------------------

def generate_intervals(
    model,
    test_df,
    features,
    quantile
):

    X_test = test_df[features]

    predictions = model.predict(X_test)

    lower = predictions - quantile
    upper = predictions + quantile

    # RUL cannot be negative.
    lower = np.maximum(lower, 0)

    result = test_df[
        ["unit_id", "cycle", "RUL"]
    ].copy()

    result["predicted_RUL"] = predictions
    result["lower_bound"] = lower
    result["upper_bound"] = upper

    result["interval_width"] = (
        upper - lower
    )

    result["covered"] = (
        (result["RUL"] >= result["lower_bound"])
        &
        (result["RUL"] <= result["upper_bound"])
    )

    return result


# --------------------------------------------------
# 7. ENGINE-LEVEL EVALUATION
# --------------------------------------------------

def evaluate_intervals(result):

    engine_results = []

    for engine_id, group in result.groupby("unit_id"):

        engine_results.append({

            "unit_id": int(engine_id),

            "engine_covered": bool(
                group["covered"].all()
            ),

            "row_coverage": float(
                group["covered"].mean()
            ),

            "mean_interval_width": float(
                group["interval_width"].mean()
            ),

            "max_interval_width": float(
                group["interval_width"].max()
            )
        })

    engine_df = pd.DataFrame(engine_results)

    engine_coverage = float(
        engine_df["engine_covered"].mean()
    )

    row_coverage = float(
        result["covered"].mean()
    )

    mean_width = float(
        result["interval_width"].mean()
    )

    mae = mean_absolute_error(
        result["RUL"],
        result["predicted_RUL"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            result["RUL"],
            result["predicted_RUL"]
        )
    )

    summary = {
        "target_coverage": 1 - ALPHA,
        "actual_engine_coverage": engine_coverage,
        "actual_row_coverage": row_coverage,
        "mean_interval_width": mean_width,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "test_engine_count": int(len(engine_df))
    }

    return summary, engine_df


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def run_conformal_prediction():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df, features, model = load_inputs()

    calibration_df, test_df = split_engines(df)

    # Calculate one calibration score per engine.
    calibration_scores = calculate_calibration_scores(
        model,
        calibration_df,
        features
    )

    quantile = calculate_quantile(
        calibration_scores.values,
        ALPHA
    )

    # Generate intervals on unseen test engines.
    result = generate_intervals(
        model,
        test_df,
        features,
        quantile
    )

    summary, engine_results = evaluate_intervals(
        result
    )

    summary["calibration_engine_count"] = int(
        len(calibration_scores)
    )

    summary["conformal_quantile"] = quantile

    summary["calibration_method"] = (
        "Engine-level maximum absolute residual"
    )

    summary["model"] = "Robust Random Forest"

    # Save prediction-level results.
    result.to_csv(
        REPORT_DIR / "conformal_predictions.csv",
        index=False
    )

    # Save engine-level results.
    engine_results.to_csv(
        REPORT_DIR / "conformal_engine_results.csv",
        index=False
    )

    # Save summary.
    with open(
        REPORT_DIR / "conformal_summary.json",
        "w"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False
        )

    print("\n" + "=" * 60)
    print("CONFORMAL PREDICTION RESULTS")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nReports saved successfully.")
    print("Day 10 experiment completed.")


if __name__ == "__main__":
    run_conformal_prediction()
