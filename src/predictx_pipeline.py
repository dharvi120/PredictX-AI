
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TEST_FILE = RAW / "test_FD001.txt"
TRAIN_FILE = RAW / "train_FD001.txt"

FEATURE_FILE = PROCESSED / "feature_columns.json"
MODEL_FILE = MODELS / "robust_random_forest.pkl"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)


# ==========================================
# 2. LOAD INPUTS
# ==========================================

def load_inputs():

    for path in [
        TEST_FILE,
        TRAIN_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    test = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None
    ).dropna(axis=1, how="all")

    train = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None
    ).dropna(axis=1, how="all")

    test.columns = COLUMNS
    train.columns = COLUMNS

    with open(FEATURE_FILE, "r") as f:
        features = json.load(f)

    model = joblib.load(MODEL_FILE)

    sensor_medians = train[SENSORS].median()

    return test, features, model, sensor_medians


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

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
                    window=5,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{sensor}_rolling_std"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    window=5,
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


# ==========================================
# 4. SENSOR HEALTH INDICATORS
# ==========================================

def calculate_sensor_health(
    engine_data,
    sensor_medians
):

    health_results = []

    for sensor in SENSORS:

        values = engine_data[sensor].astype(float)

        missing_rate = float(values.isna().mean())

        valid = values.dropna()

        if len(valid) < 2:
            health_results.append({
                "sensor": sensor,
                "health_score": 0.0,
                "status": "Insufficient data"
            })
            continue

        stuck_rate = float(
            valid.diff().eq(0).mean()
        )

        median = valid.median()

        mad = np.median(
            np.abs(valid.to_numpy() - median)
        )

        if mad > 0:
            robust_z = (
                0.6745 * (valid - median).abs() / mad
            )

            outlier_rate = float(
                (robust_z > 3.5).mean()
            )
        else:
            outlier_rate = 0.0

        changes = valid.diff().dropna()

        signal_std = float(valid.std())
        change_std = float(changes.std())

        noise_ratio = (
            change_std / signal_std
            if signal_std > 0
            else 0.0
        )

        noise_penalty = min(noise_ratio / 2, 1)

        score = 100 * (
            1
            - 0.35 * missing_rate
            - 0.30 * stuck_rate
            - 0.20 * min(outlier_rate, 1)
            - 0.15 * noise_penalty
        )

        score = float(np.clip(score, 0, 100))

        if score >= 85:
            status = "Healthy"
        elif score >= 65:
            status = "Warning"
        else:
            status = "Critical"

        health_results.append({
            "sensor": sensor,
            "health_score": score,
            "status": status
        })

    return health_results


# ==========================================
# 5. PREDICTION PIPELINE
# ==========================================

def run_pipeline():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading PredictX AI components...")

    test, features, model, sensor_medians = load_inputs()

    print("Generating engineered features...")

    engineered = create_features(test)

    # One latest observation per engine.
    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    X = final_rows[features].copy()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Invalid values detected in model features."
        )

    print("Predicting remaining useful life...")

    predictions = model.predict(X)

    output = []

    for index, row in final_rows.iterrows():

        engine_id = int(row["unit_id"])

        engine_data = test[
            test["unit_id"] == engine_id
        ].sort_values("cycle")

        health = calculate_sensor_health(
            engine_data,
            sensor_medians
        )

        health_scores = [
            item["health_score"]
            for item in health
        ]

        average_health = float(
            np.mean(health_scores)
        )

        unhealthy_sensors = [
            item["sensor"]
            for item in health
            if item["status"] in ["Warning", "Critical"]
        ]

        output.append({
            "unit_id": engine_id,
            "last_cycle": int(row["cycle"]),
            "predicted_rul": float(predictions[index]),
            "average_sensor_health": average_health,
            "unhealthy_sensor_count": len(unhealthy_sensors),
            "unhealthy_sensors": ", ".join(unhealthy_sensors)
        })

    result_df = pd.DataFrame(output)

    result_df.to_csv(
        REPORTS / "predictx_pipeline_output.csv",
        index=False
    )

    summary = {
        "total_engines": int(len(result_df)),
        "average_predicted_rul": float(
            result_df["predicted_rul"].mean()
        ),
        "average_sensor_health": float(
            result_df["average_sensor_health"].mean()
        ),
        "engines_with_unhealthy_sensors": int(
            (
                result_df["unhealthy_sensor_count"] > 0
            ).sum()
        ),
        "model": "Robust Random Forest",
        "health_method": "Experimental rule-based indicators"
    }

    with open(
        REPORTS / "predictx_pipeline_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n========== PREDICTX AI PIPELINE ==========")

    print(result_df.head(10).to_string(index=False))

    print("\n========== SYSTEM SUMMARY ==========")

    print(json.dumps(summary, indent=4))

    print("\nReports saved successfully.")


if __name__ == "__main__":
    run_pipeline()
