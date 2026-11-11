
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

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "model_train.csv"
VALIDATION_FILE = DATA_DIR / "model_validation.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.json"

MODEL_FILE = MODEL_DIR / "random_forest.pkl"


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

def load_inputs():

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    with open(FEATURE_FILE, "r") as file:
        feature_columns = json.load(file)

    model = joblib.load(MODEL_FILE)

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()

    y_validation = validation_df["RUL"].copy()

    return X_train, X_validation, y_validation, feature_columns, model


# --------------------------------------------------
# 3. IDENTIFY SENSOR FEATURE GROUPS
# --------------------------------------------------

def identify_sensor_groups(feature_columns):

    groups = {}

    # Find original sensor columns, e.g. sensor_2.
    sensor_columns = [
        col for col in feature_columns
        if col.startswith("sensor_")
        and col.count("_") == 1
    ]

    for sensor in sensor_columns:

        # Include the original sensor and its derived features.
        related_columns = [
            col for col in feature_columns
            if col == sensor or col.startswith(sensor + "_")
        ]

        groups[sensor] = related_columns

    if not groups:
        raise ValueError(
            "No sensor columns found. Check feature column names."
        )

    print("Sensor groups identified:", len(groups))

    return groups


# --------------------------------------------------
# 4. SENSOR CORRUPTION SIMULATORS
# --------------------------------------------------

def simulate_missing_values(X, groups, rate=0.10, seed=42):

    corrupted = X.copy()
    rng = np.random.default_rng(seed)

    affected = []

    for sensor, columns in groups.items():

        for col in columns:

            mask = rng.random(len(corrupted)) < rate
            corrupted.loc[mask, col] = np.nan

        affected.append(sensor)

    return corrupted, affected


def simulate_sensor_noise(
    X,
    groups,
    training_std,
    noise_level=0.20,
    seed=42
):

    corrupted = X.copy()
    rng = np.random.default_rng(seed)

    affected = []

    for sensor, columns in groups.items():

        for col in columns:

            scale = training_std[col]

            noise = rng.normal(
                loc=0,
                scale=noise_level * scale,
                size=len(corrupted)
            )

            corrupted[col] = corrupted[col] + noise

        affected.append(sensor)

    return corrupted, affected


def simulate_sensor_dropout(
    X,
    groups,
    training_medians,
    dropout_count=2,
    seed=42
):

    corrupted = X.copy()
    rng = np.random.default_rng(seed)

    available_sensors = list(groups.keys())

    selected = rng.choice(
        available_sensors,
        size=min(dropout_count, len(available_sensors)),
        replace=False
    )

    for sensor in selected:

        for col in groups[sensor]:
            corrupted[col] = training_medians[col]

    return corrupted, list(selected)


# --------------------------------------------------
# 5. HANDLE MISSING VALUES
# --------------------------------------------------

def impute_missing_values(X, training_medians):

    result = X.copy()

    result = result.fillna(training_medians)

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite values remain after imputation."
        )

    return result


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

def evaluate(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }


# --------------------------------------------------
# 7. SENSOR RELIABILITY SUMMARY
# --------------------------------------------------

def calculate_reliability(
    scenario_name,
    X_before_imputation,
    groups,
    affected_sensors
):

    records = []

    for sensor, columns in groups.items():

        values = X_before_imputation[columns]

        missing_rate = float(
            values.isna().sum().sum() / values.size
        )

        # This is a controlled-experiment score.
        # It uses known injected corruption, not an
        # independent real-world fault detector.
        if sensor in affected_sensors:
            score = 0.0
        else:
            score = 1.0

        records.append({
            "scenario": scenario_name,
            "sensor": sensor,
            "missing_rate": missing_rate,
            "controlled_reliability_score": score
        })

    return records


# --------------------------------------------------
# 8. RUN EXPERIMENTS
# --------------------------------------------------

def run_experiments():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        X_validation,
        y_validation,
        feature_columns,
        model
    ) = load_inputs()

    groups = identify_sensor_groups(feature_columns)

    # Training-only statistics prevent validation leakage.
    training_medians = X_train.median()
    training_std = X_train.std().fillna(0)

    # Clean baseline.
    scenarios = {
        "Clean": (
            X_validation.copy(),
            []
        )
    }

    # Missing readings.
    missing_data, missing_sensors = simulate_missing_values(
        X_validation,
        groups,
        rate=0.10
    )

    scenarios["Missing_10_Percent"] = (
        missing_data,
        missing_sensors
    )

    # Additive sensor noise.
    noisy_data, noisy_sensors = simulate_sensor_noise(
        X_validation,
        groups,
        training_std,
        noise_level=0.20
    )

    scenarios["Noise_20_Percent"] = (
        noisy_data,
        noisy_sensors
    )

    # Two sensors completely stuck at their training median.
    dropout_data, dropout_sensors = simulate_sensor_dropout(
        X_validation,
        groups,
        training_medians,
        dropout_count=2
    )

    scenarios["Dropout_2_Sensors"] = (
        dropout_data,
        dropout_sensors
    )

    results = []
    reliability_records = []

    for scenario_name, (corrupted_X, affected) in scenarios.items():

        print("\nRunning scenario:", scenario_name)

        reliability_records.extend(
            calculate_reliability(
                scenario_name,
                corrupted_X,
                groups,
                affected
            )
        )

        # Random Forest needs finite numeric input.
        prepared_X = impute_missing_values(
            corrupted_X,
            training_medians
        )

        metrics = evaluate(
            model,
            prepared_X,
            y_validation
        )

        results.append({
            "Scenario": scenario_name,
            **metrics
        })

        print("MAE:", round(metrics["MAE"], 4))
        print("RMSE:", round(metrics["RMSE"], 4))

    # Save performance report.
    results_df = pd.DataFrame(results)

    baseline_mae = results_df.loc[
        results_df["Scenario"] == "Clean",
        "MAE"
    ].iloc[0]

    results_df["MAE_Degradation"] = (
        results_df["MAE"] - baseline_mae
    )

    results_df["MAE_Degradation_Percent"] = (
        results_df["MAE_Degradation"] / baseline_mae
    ) * 100

    results_df.to_csv(
        REPORT_DIR / "sensor_robustness_results.csv",
        index=False
    )

    # Save sensor-level controlled experiment report.
    reliability_df = pd.DataFrame(reliability_records)

    reliability_df.to_csv(
        REPORT_DIR / "sensor_reliability_report.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("SENSOR ROBUSTNESS COMPARISON")
    print("=" * 55)

    print(results_df.to_string(index=False))

    print("\nReports saved successfully.")
    print("Day 8 experiment completed.")


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

if __name__ == "__main__":
    run_experiments()
