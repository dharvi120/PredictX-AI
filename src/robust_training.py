
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

BASELINE_FILE = MODEL_DIR / "random_forest.pkl"

ROBUST_MODEL_FILE = MODEL_DIR / "robust_random_forest.pkl"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        FEATURE_FILE,
        BASELINE_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    with open(FEATURE_FILE, "r") as file:
        features = json.load(file)

    X_train = train_df[features].copy()
    y_train = train_df["RUL"].copy()

    X_validation = validation_df[features].copy()
    y_validation = validation_df["RUL"].copy()

    baseline_model = joblib.load(BASELINE_FILE)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)
    print("Features:", len(features))

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        baseline_model,
        features
    )


# --------------------------------------------------
# 3. IDENTIFY SENSOR FEATURE GROUPS
# --------------------------------------------------

def identify_sensor_groups(features):

    groups = {}

    for column in features:

        parts = column.split("_")

        if len(parts) < 2 or parts[0] != "sensor":
            continue

        sensor_name = "_".join(parts[:2])

        groups.setdefault(sensor_name, []).append(column)

    if not groups:
        raise ValueError("No sensor feature groups identified.")

    print("\nSensor groups:", len(groups))

    return groups


# --------------------------------------------------
# 4. CORRUPTION FUNCTIONS
# --------------------------------------------------

def add_missing_values(X, groups, rate=0.10, seed=42):

    result = X.copy()
    rng = np.random.default_rng(seed)

    for sensor, columns in groups.items():

        for column in columns:

            mask = rng.random(len(result)) < rate
            result.loc[mask, column] = np.nan

    return result


def add_noise(
    X,
    groups,
    training_std,
    noise_level=0.20,
    seed=42
):

    result = X.copy()
    rng = np.random.default_rng(seed)

    for sensor, columns in groups.items():

        for column in columns:

            scale = training_std[column]

            noise = rng.normal(
                0,
                noise_level * scale,
                size=len(result)
            )

            result[column] = result[column] + noise

    return result


def add_sensor_dropout(
    X,
    groups,
    training_medians,
    dropout_count=2,
    seed=42
):

    result = X.copy()
    rng = np.random.default_rng(seed)

    sensors = list(groups.keys())

    selected = rng.choice(
        sensors,
        size=min(dropout_count, len(sensors)),
        replace=False
    )

    for sensor in selected:

        for column in groups[sensor]:
            result[column] = training_medians[column]

    return result


# --------------------------------------------------
# 5. HANDLE MISSING VALUES
# --------------------------------------------------

def impute_data(X, training_medians):

    result = X.fillna(training_medians)

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite values remain.")

    return result


# --------------------------------------------------
# 6. CREATE AUGMENTED TRAINING DATA
# --------------------------------------------------

def create_augmented_data(
    X_train,
    y_train,
    groups
):

    training_medians = X_train.median()
    training_std = X_train.std().fillna(0)

    # Original clean observations.
    clean_X = X_train.copy()

    # Corrupted copy 1: Missing readings.
    missing_X = add_missing_values(
        X_train,
        groups,
        rate=0.10,
        seed=101
    )

    # Corrupted copy 2: Sensor noise.
    noisy_X = add_noise(
        X_train,
        groups,
        training_std,
        noise_level=0.20,
        seed=202
    )

    # Corrupted copy 3: Sensor dropout.
    dropout_X = add_sensor_dropout(
        X_train,
        groups,
        training_medians,
        dropout_count=2,
        seed=303
    )

    # Impute missing values using training-only statistics.
    clean_X = impute_data(clean_X, training_medians)
    missing_X = impute_data(missing_X, training_medians)
    noisy_X = impute_data(noisy_X, training_medians)
    dropout_X = impute_data(dropout_X, training_medians)

    # Combine all four datasets.
    X_augmented = pd.concat(
        [
            clean_X,
            missing_X,
            noisy_X,
            dropout_X
        ],
        ignore_index=True
    )

    y_augmented = pd.concat(
        [y_train, y_train, y_train, y_train],
        ignore_index=True
    )

    print("\nAugmentation completed.")
    print("Original training rows:", len(X_train))
    print("Augmented training rows:", len(X_augmented))

    return (
        X_augmented,
        y_augmented,
        training_medians,
        training_std
    )


# --------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------

def evaluate_model(model, X, y):

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
# 8. RUN VALIDATION SCENARIOS
# --------------------------------------------------

def evaluate_scenarios(
    model,
    X_validation,
    y_validation,
    groups,
    training_medians,
    training_std
):

    scenarios = {
        "Clean": X_validation.copy(),

        "Missing_10_Percent": add_missing_values(
            X_validation,
            groups,
            rate=0.10,
            seed=42
        ),

        "Noise_20_Percent": add_noise(
            X_validation,
            groups,
            training_std,
            noise_level=0.20,
            seed=42
        ),

        "Dropout_2_Sensors": add_sensor_dropout(
            X_validation,
            groups,
            training_medians,
            dropout_count=2,
            seed=42
        )
    }

    results = []

    for scenario_name, corrupted_X in scenarios.items():

        prepared_X = impute_data(
            corrupted_X,
            training_medians
        )

        metrics = evaluate_model(
            model,
            prepared_X,
            y_validation
        )

        results.append({
            "Scenario": scenario_name,
            **metrics
        })

    return results


# --------------------------------------------------
# 9. MAIN EXPERIMENT
# --------------------------------------------------

def run_robust_training():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        baseline_model,
        features
    ) = load_data()

    groups = identify_sensor_groups(features)

    (
        X_augmented,
        y_augmented,
        training_medians,
        training_std
    ) = create_augmented_data(
        X_train,
        y_train,
        groups
    )

    # Train a separate robust Random Forest.
    print("\nTraining robust Random Forest...")

    robust_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    robust_model.fit(
        X_augmented,
        y_augmented
    )

    joblib.dump(
        robust_model,
        ROBUST_MODEL_FILE
    )

    print("Robust model saved:", ROBUST_MODEL_FILE)

    # Evaluate both models under identical scenarios.
    baseline_results = evaluate_scenarios(
        baseline_model,
        X_validation,
        y_validation,
        groups,
        training_medians,
        training_std
    )

    robust_results = evaluate_scenarios(
        robust_model,
        X_validation,
        y_validation,
        groups,
        training_medians,
        training_std
    )

    baseline_df = pd.DataFrame(baseline_results)
    robust_df = pd.DataFrame(robust_results)

    baseline_df["Model"] = "Original Random Forest"
    robust_df["Model"] = "Robust Random Forest"

    comparison = pd.concat(
        [baseline_df, robust_df],
        ignore_index=True
    )

    # Calculate improvement relative to original model.
    baseline_mae = (
        baseline_df.set_index("Scenario")["MAE"]
    )

    comparison["MAE_Improvement_Percent"] = (
        comparison.apply(
            lambda row: (
                (baseline_mae[row["Scenario"]] - row["MAE"])
                / baseline_mae[row["Scenario"]]
            ) * 100,
            axis=1
        )
    )

    comparison_path = (
        REPORT_DIR / "robust_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("ORIGINAL VS ROBUST RANDOM FOREST")
    print("=" * 70)

    print(
        comparison.round(4).to_string(index=False)
    )

    print("\nReport saved:", comparison_path)
    print("Day 9 experiment completed.")


if __name__ == "__main__":
    run_robust_training()
