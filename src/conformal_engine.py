
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_RAW = RAW / "train_FD001.txt"
TEST_RAW = RAW / "test_FD001.txt"

TRAIN_MODEL = PROCESSED / "model_train.csv"
FEATURE_FILE = PROCESSED / "feature_columns.json"

MODEL_FILE = MODELS / "robust_random_forest.pkl"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)

WINDOW = 5
ALPHA = 0.10
CALIBRATION_FRACTION = 0.20
SEED = 42


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
            .diff().fillna(0)
        )

    return df


# ==================================================
# 3. LOAD TRAINING DATA
# ==================================================

def load_training_data():

    train = pd.read_csv(
        TRAIN_RAW,
        sep=r"\s+",
        header=None
    ).dropna(axis=1, how="all")

    train.columns = COLUMNS

    with open(FEATURE_FILE, "r") as f:
        features = json.load(f)

    engineered = create_features(train)

    # Training RUL label
    max_cycles = (
        train.groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    engineered = engineered.join(
        max_cycles,
        on="unit_id"
    )

    engineered["RUL"] = (
        engineered["max_cycle"] - engineered["cycle"]
    )

    # Match preprocessing: remove constant features
    # using training data only.
    model_train = pd.read_csv(TRAIN_MODEL)

    feature_medians = model_train[features].median()

    return engineered, features, feature_medians


# ==================================================
# 4. ENGINE-WISE CALIBRATION SPLIT
# ==================================================

def split_engines(data):

    rng = np.random.default_rng(SEED)

    engine_ids = np.sort(
        data["unit_id"].unique()
    )

    rng.shuffle(engine_ids)

    calibration_count = math.ceil(
        len(engine_ids) * CALIBRATION_FRACTION
    )

    calibration_ids = set(
        engine_ids[:calibration_count]
    )

    calibration_data = data[
        data["unit_id"].isin(calibration_ids)
    ].copy()

    return calibration_data, len(calibration_ids)


# ==================================================
# 5. CALIBRATE CONFORMAL INTERVAL
# ==================================================

def calibrate_model(
    model,
    calibration_data,
    features,
    feature_medians
):

    X_cal = calibration_data[features].copy()
    X_cal = X_cal.fillna(feature_medians)

    if not np.isfinite(X_cal.to_numpy(dtype=float)).all():
        raise ValueError("Invalid calibration features.")

    predictions = model.predict(X_cal)

    calibration_data = calibration_data.copy()

    calibration_data["prediction"] = predictions

    calibration_data["absolute_error"] = (
        calibration_data["RUL"]
        - calibration_data["prediction"]
    ).abs()

    # One nonconformity score per engine:
    # maximum absolute residual over its trajectory.
    engine_scores = (
        calibration_data.groupby("unit_id")["absolute_error"]
        .max()
        .to_numpy()
    )

    n = len(engine_scores)

    rank = math.ceil(
        (n + 1) * (1 - ALPHA)
    )

    if rank > n:
        raise ValueError(
            "Insufficient calibration engines for the "
            "requested finite-sample conformal quantile."
        )

    quantile = float(
        np.sort(engine_scores)[rank - 1]
    )

    return quantile, n


# ==================================================
# 6. EVALUATE ON NASA TEST ENGINES
# ==================================================

def evaluate_test(
    model,
    test,
    features,
    feature_medians,
    quantile
):

    engineered = create_features(test)

    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    X_test = final_rows[features].copy()
    X_test = X_test.fillna(feature_medians)

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Invalid test features.")

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "unit_id": final_rows["unit_id"],
        "last_cycle": final_rows["cycle"],
        "predicted_rul": predictions
    })

    results["lower_rul"] = np.maximum(
        0,
        results["predicted_rul"] - quantile
    )

    results["upper_rul"] = (
        results["predicted_rul"] + quantile
    )

    results["interval_width"] = (
        results["upper_rul"] - results["lower_rul"]
    )

    results["uncertainty_width"] = results["interval_width"]

    results["uncertainty_level"] = np.where(
        results["interval_width"] > 150,
        "High",
        np.where(
            results["interval_width"] > 80,
            "Medium",
            "Low"
        )
    )

    return results


# ==================================================
# 7. MAIN
# ==================================================

def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading model...")

    for path in [
        TRAIN_RAW,
        TEST_RAW,
        TRAIN_MODEL,
        FEATURE_FILE,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    model = joblib.load(MODEL_FILE)

    print("Preparing training trajectories...")

    train_data, features, feature_medians = (
        load_training_data()
    )

    calibration_data, engine_count = split_engines(
        train_data
    )

    print(
        "Calibration engines:",
        engine_count
    )

    print("Calibrating uncertainty...")

    quantile, n = calibrate_model(
        model,
        calibration_data,
        features,
        feature_medians
    )

    print("Conformal quantile:", round(quantile, 4))

    test = pd.read_csv(
        TEST_RAW,
        sep=r"\s+",
        header=None
    ).dropna(axis=1, how="all")

    test.columns = COLUMNS

    results = evaluate_test(
        model,
        test,
        features,
        feature_medians,
        quantile
    )

    results.to_csv(
        REPORTS / "conformal_test_predictions.csv",
        index=False
    )

    summary = {
        "model": "Robust Random Forest",
        "calibration_method": "Engine-level maximum absolute residual",
        "calibration_engines": int(n),
        "target_coverage": 1 - ALPHA,
        "alpha": ALPHA,
        "conformal_quantile": quantile,
        "mean_interval_width": float(
            results["interval_width"].mean()
        ),
        "minimum_interval_width": float(
            results["interval_width"].min()
        ),
        "maximum_interval_width": float(
            results["interval_width"].max()
        ),
        "test_engines": int(len(results))
    }

    with open(
        REPORTS / "conformal_test_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n========== CONFORMAL PREDICTION ==========")

    print(json.dumps(summary, indent=4))

    print("\nFirst five predictions:")

    print(
        results.head(5).to_string(index=False)
    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()
