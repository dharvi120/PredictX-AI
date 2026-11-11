
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TEST_FILE = RAW / "test_FD001.txt"
RUL_FILE = RAW / "RUL_FD001.txt"

VALIDATION_FILE = PROCESSED / "model_validation.csv"
FEATURE_FILE = PROCESSED / "feature_columns.json"

MODEL_FILE = MODELS / "robust_random_forest.pkl"

ALPHA = 0.10
WINDOW = 5

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)


# ==================================================
# 2. FEATURE ENGINEERING
# ==================================================

def create_features(df):

    df = df.copy()

    df = df.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    for sensor in SENSORS:

        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    WINDOW,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{sensor}_rolling_std"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    WINDOW,
                    min_periods=2
                ).std()
            ).fillna(0)
        )

        df[f"{sensor}_change"] = (
            df.groupby("unit_id")[sensor]
            .diff()
            .fillna(0)
        )

    return df


# ==================================================
# 3. LOAD DATA
# ==================================================

def load_inputs():

    required = [
        TEST_FILE,
        RUL_FILE,
        VALIDATION_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    validation = pd.read_csv(VALIDATION_FILE)

    with open(FEATURE_FILE, "r") as f:
        features = json.load(f)

    model = joblib.load(MODEL_FILE)

    test = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None
    ).dropna(axis=1, how="all")

    if test.shape[1] != 26:
        raise ValueError("Expected 26 columns in NASA test file.")

    test.columns = COLUMNS

    true_rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    ).iloc[:, 0].to_numpy()

    return validation, test, true_rul, features, model


# ==================================================
# 4. CALIBRATION ON HELD-OUT VALIDATION ENGINES
# ==================================================

def calibrate(validation, features, model):

    required_columns = (
        features + ["unit_id", "RUL"]
    )

    missing = [
        col for col in required_columns
        if col not in validation.columns
    ]

    if missing:
        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    X_cal = validation[features].copy()
    y_cal = validation["RUL"].to_numpy()

    if not np.isfinite(X_cal.to_numpy(dtype=float)).all():
        raise ValueError("Invalid calibration features.")

    predictions = model.predict(X_cal)

    calibration = pd.DataFrame({
        "unit_id": validation["unit_id"].to_numpy(),
        "absolute_error": np.abs(y_cal - predictions)
    })

    # One maximum absolute residual per engine.
    engine_scores = (
        calibration.groupby("unit_id")["absolute_error"]
        .max()
        .to_numpy()
    )

    n = len(engine_scores)

    if n == 0:
        raise ValueError("No calibration engines found.")

    rank = math.ceil((n + 1) * (1 - ALPHA))

    if rank > n:
        raise ValueError(
            "Not enough calibration engines for the requested "
            "finite-sample conformal quantile."
        )

    quantile = float(
        np.sort(engine_scores)[rank - 1]
    )

    return quantile, n, engine_scores


# ==================================================
# 5. PREPARE OFFICIAL TEST ENGINES
# ==================================================

def prepare_test_features(test, features):

    engineered = create_features(test)

    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    missing = [
        col for col in features
        if col not in final_rows.columns
    ]

    if missing:
        raise ValueError(
            f"Missing test features: {missing}"
        )

    X_test = final_rows[features].copy()

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Invalid test features.")

    return final_rows, X_test


# ==================================================
# 6. EVALUATE COVERAGE
# ==================================================

def evaluate(
    model,
    final_rows,
    X_test,
    true_rul,
    quantile
):

    if len(final_rows) != len(true_rul):
        raise ValueError("Test engine and RUL counts differ.")

    predictions = model.predict(X_test)

    lower = np.maximum(
        0,
        predictions - quantile
    )

    upper = predictions + quantile

    covered = (
        (true_rul >= lower)
        & (true_rul <= upper)
    )

    results = pd.DataFrame({
        "unit_id": final_rows["unit_id"],
        "last_cycle": final_rows["cycle"],
        "true_rul": true_rul,
        "predicted_rul": predictions,
        "lower_rul": lower,
        "upper_rul": upper,
        "interval_width": upper - lower,
        "covered": covered
    })

    mae = mean_absolute_error(
        true_rul,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_rul,
            predictions
        )
    )

    coverage = float(covered.mean())

    summary = {
        "model": "Robust Random Forest",
        "calibration_method": "Engine-level maximum absolute residual",
        "target_coverage": 1 - ALPHA,
        "empirical_test_coverage": coverage,
        "covered_engines": int(covered.sum()),
        "total_test_engines": int(len(covered)),
        "calibration_engine_count": None,
        "conformal_quantile": float(quantile),
        "mean_interval_width": float(
            results["interval_width"].mean()
        ),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }

    return results, summary


# ==================================================
# 7. MAIN
# ==================================================

def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading validation data, test data and model...")

    validation, test, true_rul, features, model = load_inputs()

    print("Validation rows:", len(validation))
    print("Validation engines:", validation["unit_id"].nunique())
    print("Test engines:", test["unit_id"].nunique())

    print("\nCalibrating using validation engines...")

    quantile, n, scores = calibrate(
        validation,
        features,
        model
    )

    print("Calibration engines:", n)
    print("Conformal quantile:", round(quantile, 4))

    print("\nPreparing official NASA test features...")

    final_rows, X_test = prepare_test_features(
        test,
        features
    )

    results, summary = evaluate(
        model,
        final_rows,
        X_test,
        true_rul,
        quantile
    )

    summary["calibration_engine_count"] = int(n)

    results.to_csv(
        REPORTS / "conformal_coverage_results.csv",
        index=False
    )

    with open(
        REPORTS / "conformal_coverage_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n========== CONFORMAL COVERAGE RESULTS ==========")

    print(json.dumps(summary, indent=4))

    print("\nFirst five engine results:")

    print(
        results.head(5).to_string(index=False)
    )

    print("\nUncovered engines:")

    print(
        results.loc[
            ~results["covered"],
            [
                "unit_id",
                "true_rul",
                "predicted_rul",
                "lower_rul",
                "upper_rul"
            ]
        ].to_string(index=False)
    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()
