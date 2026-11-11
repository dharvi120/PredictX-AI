
from pathlib import Path
from itertools import combinations
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_FILE = RAW / "train_FD001.txt"
TEST_FILE = RAW / "test_FD001.txt"
RUL_FILE = RAW / "RUL_FD001.txt"

FEATURE_FILE = PROCESSED / "feature_columns.json"
SHAP_FILE = REPORTS / "shap_feature_importance.csv"

BASELINE_FILE = MODELS / "random_forest.pkl"
ROBUST_FILE = MODELS / "robust_random_forest.pkl"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)

WINDOW = 5


# ==================================================
# 2. LOAD DATA
# ==================================================

def read_nasa_file(path):

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 26:
        raise ValueError(
            f"Expected 26 columns in {path.name}; "
            f"received {df.shape[1]}"
        )

    df.columns = COLUMNS

    return df


def load_inputs():

    required = [
        TRAIN_FILE,
        TEST_FILE,
        RUL_FILE,
        FEATURE_FILE,
        BASELINE_FILE,
        ROBUST_FILE
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    train = read_nasa_file(TRAIN_FILE)
    test = read_nasa_file(TEST_FILE)

    true_rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    ).iloc[:, 0].to_numpy()

    with open(FEATURE_FILE, "r") as file:
        features = json.load(file)

    baseline = joblib.load(BASELINE_FILE)
    robust = joblib.load(ROBUST_FILE)

    if len(test["unit_id"].unique()) != len(true_rul):
        raise ValueError("Test engine and RUL label counts differ.")

    # Training-only sensor medians.
    sensor_medians = train[SENSORS].median()

    return (
        test,
        true_rul,
        features,
        baseline,
        robust,
        sensor_medians
    )


# ==================================================
# 3. FEATURE ENGINEERING
# ==================================================

def create_features(data):

    df = data.copy()

    df = df.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    for sensor in SENSORS:

        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    window=WINDOW,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{sensor}_rolling_std"] = (
            df.groupby("unit_id")[sensor]
            .transform(
                lambda s: s.rolling(
                    window=WINDOW,
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
# 4. PREPARE FEATURES AFTER SENSOR FAILURE
# ==================================================

def prepare_features(
    test_data,
    failed_sensors,
    features,
    sensor_medians
):

    corrupted = test_data.copy(deep=True)

    # A failed sensor is stuck at its training median
    for sensor in failed_sensors:
        corrupted[sensor] = sensor_medians[sensor]

    engineered = create_features(corrupted)

    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    missing = [
        feature for feature in features
        if feature not in final_rows.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = final_rows[features].copy()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Invalid values in test features.")

    return X


# ==================================================
# 5. MODEL METRICS
# ==================================================

def calculate_metrics(y_true, predictions):

    return {
        "MAE": float(
            mean_absolute_error(y_true, predictions)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, predictions)
            )
        )
    }


# ==================================================
# 6. RUN ONE SENSOR FAILURE EXPERIMENT
# ==================================================

def evaluate_failure(
    test,
    y_true,
    failed_sensors,
    features,
    sensor_medians,
    baseline,
    robust
):

    X = prepare_features(
        test,
        failed_sensors,
        features,
        sensor_medians
    )

    baseline_predictions = baseline.predict(X)
    robust_predictions = robust.predict(X)

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_predictions
    )

    robust_metrics = calculate_metrics(
        y_true,
        robust_predictions
    )

    sensor_label = "+".join(failed_sensors)

    return [
        {
            "failure_type": (
                "Individual" if len(failed_sensors) == 1
                else "Pair"
            ),
            "failed_sensors": sensor_label,
            "sensor_count": len(failed_sensors),
            "model": "Baseline Random Forest",
            **baseline_metrics
        },
        {
            "failure_type": (
                "Individual" if len(failed_sensors) == 1
                else "Pair"
            ),
            "failed_sensors": sensor_label,
            "sensor_count": len(failed_sensors),
            "model": "Robust Random Forest",
            **robust_metrics
        }
    ]


# ==================================================
# 7. SHAP IMPORTANCE MAPPING
# ==================================================

def create_shap_sensor_ranking():

    if not SHAP_FILE.exists():
        print(
            "\nWARNING: SHAP importance file not found."
        )
        print(
            "Sensor failure analysis will continue, "
            "but SHAP comparison will be skipped."
        )
        return None

    shap_df = pd.read_csv(SHAP_FILE)

    required_columns = {
        "feature",
        "mean_absolute_shap"
    }

    if not required_columns.issubset(shap_df.columns):
        print(
            "WARNING: SHAP CSV has unexpected columns. "
            "Skipping SHAP comparison."
        )
        return None

    rows = []

    for sensor in SENSORS:

        related_features = [
            feature for feature in shap_df["feature"]
            if feature == sensor
            or feature.startswith(sensor + "_")
        ]

        total_importance = shap_df.loc[
            shap_df["feature"].isin(related_features),
            "mean_absolute_shap"
        ].sum()

        rows.append({
            "sensor": sensor,
            "shap_importance": float(total_importance)
        })

    ranking = pd.DataFrame(rows).sort_values(
        "shap_importance",
        ascending=False
    )

    return ranking


# ==================================================
# 8. VISUALIZATION
# ==================================================

def generate_failure_plot(individual, pairs):

    plt.figure(figsize=(12, 7))

    plt.bar(
        individual["failed_sensors"],
        individual["baseline_degradation"]
        if "baseline_degradation" in individual.columns
        else individual["baseline_MAE"]
    )

    plt.xlabel("Sensor")
    plt.ylabel("Baseline MAE degradation (cycles)")
    plt.title("Individual Sensor Failure Impact")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        REPORTS / "sensor_failure_analysis.png",
        dpi=300
    )

    plt.close()

    top_pairs = pairs.nlargest(
        10,
        "baseline_degradation"
    ).sort_values("baseline_degradation")

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_pairs["failed_sensors"],
        top_pairs["baseline_degradation"]
    )

    plt.xlabel("Baseline MAE degradation (cycles)")
    plt.ylabel("Failed sensor pair")
    plt.title("Top 10 Most Damaging Sensor Pairs")
    plt.tight_layout()

    plt.savefig(
        REPORTS / "sensor_pair_failure_analysis.png",
        dpi=300
    )

    plt.close()


# ==================================================
# 9. MAIN EXPERIMENT
# ==================================================

def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading NASA data and models...")

    (
        test,
        y_true,
        features,
        baseline,
        robust,
        sensor_medians
    ) = load_inputs()

    print("Test engines:", test["unit_id"].nunique())
    print("Sensors:", len(SENSORS))

    # Clean baseline metrics
    print("\nCalculating clean reference metrics...")

    clean_results = evaluate_failure(
        test,
        y_true,
        [],
        features,
        sensor_medians,
        baseline,
        robust
    )

    clean_baseline_mae = clean_results[0]["MAE"]
    clean_robust_mae = clean_results[1]["MAE"]

    print("Clean baseline MAE:", clean_baseline_mae)
    print("Clean robust MAE:", clean_robust_mae)

    all_results = []

    # Individual sensor failures
    print("\nTesting 21 individual sensor failures...")

    for i, sensor in enumerate(SENSORS, start=1):

        result = evaluate_failure(
            test,
            y_true,
            [sensor],
            features,
            sensor_medians,
            baseline,
            robust
        )

        all_results.extend(result)

        print(f"Individual sensor {i}/21: {sensor}")

    # All 210 sensor pairs
    sensor_pairs = list(combinations(SENSORS, 2))

    print("\nTesting 210 sensor pairs...")

    for i, pair in enumerate(sensor_pairs, start=1):

        result = evaluate_failure(
            test,
            y_true,
            list(pair),
            features,
            sensor_medians,
            baseline,
            robust
        )

        all_results.extend(result)

        if i % 25 == 0 or i == len(sensor_pairs):
            print(
                f"Completed {i}/{len(sensor_pairs)} pairs"
            )

    # Save raw results
    results = pd.DataFrame(all_results)

    results.to_csv(
        REPORTS / "sensor_dropout_ranking.csv",
        index=False
    )

    # Separate individual failures and pairs
    individual = results[
        results["failure_type"] == "Individual"
    ].copy()

    pairs = results[
        results["failure_type"] == "Pair"
    ].copy()

    individual["baseline_degradation"] = (
        individual.apply(
            lambda row: row["MAE"] - clean_baseline_mae
            if row["model"] == "Baseline Random Forest"
            else np.nan,
            axis=1
        )
    )

    individual["robust_degradation"] = (
        individual.apply(
            lambda row: row["MAE"] - clean_robust_mae
            if row["model"] == "Robust Random Forest"
            else np.nan,
            axis=1
        )
    )

    pairs["baseline_degradation"] = (
        pairs.apply(
            lambda row: row["MAE"] - clean_baseline_mae
            if row["model"] == "Baseline Random Forest"
            else np.nan,
            axis=1
        )
    )

    pairs["robust_degradation"] = (
        pairs.apply(
            lambda row: row["MAE"] - clean_robust_mae
            if row["model"] == "Robust Random Forest"
            else np.nan,
            axis=1
        )
    )

    # Save pair results
    pairs.to_csv(
        REPORTS / "sensor_pair_dropout_results.csv",
        index=False
    )

    # Save individual ranking
    individual.to_csv(
        REPORTS / "sensor_individual_dropout_results.csv",
        index=False
    )

    # Create paired comparison table
    pair_pivot = pairs.pivot(
        index="failed_sensors",
        columns="model",
        values="MAE"
    ).reset_index()

    pair_pivot["baseline_minus_robust_MAE"] = (
        pair_pivot["Baseline Random Forest"]
        - pair_pivot["Robust Random Forest"]
    )

    pair_pivot.to_csv(
        REPORTS / "sensor_pair_model_comparison.csv",
        index=False
    )

    # SHAP comparison
    shap_ranking = create_shap_sensor_ranking()

    if shap_ranking is not None:

        individual_pivot = individual.pivot(
            index="failed_sensors",
            columns="model",
            values="MAE"
        ).reset_index()

        individual_pivot["baseline_degradation"] = (
            individual_pivot["Baseline Random Forest"]
            - clean_baseline_mae
        )

        individual_pivot["robust_degradation"] = (
            individual_pivot["Robust Random Forest"]
            - clean_robust_mae
        )

        individual_pivot["sensor"] = (
            individual_pivot["failed_sensors"]
        )

        comparison = shap_ranking.merge(
            individual_pivot,
            on="sensor",
            how="left"
        )

        comparison.to_csv(
            REPORTS / "sensor_importance_vs_failure.csv",
            index=False
        )

    # Visualizations
    generate_failure_plot(
        individual,
        pairs
    )

    # Final rankings
    baseline_individual = individual[
        individual["model"] == "Baseline Random Forest"
    ].sort_values(
        "baseline_degradation",
        ascending=False
    )

    baseline_pairs = pairs[
        pairs["model"] == "Baseline Random Forest"
    ].sort_values(
        "baseline_degradation",
        ascending=False
    )

    robust_pairs = pairs[
        pairs["model"] == "Robust Random Forest"
    ].sort_values(
        "robust_degradation",
        ascending=False
    )

    print("\n========== TOP 10 INDIVIDUAL SENSOR FAILURES ==========")

    print(
        baseline_individual[
            [
                "failed_sensors",
                "MAE",
                "baseline_degradation"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    print("\n========== TOP 10 BASELINE SENSOR PAIRS ==========")

    print(
        baseline_pairs[
            [
                "failed_sensors",
                "MAE",
                "baseline_degradation"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    print("\n========== TOP 10 ROBUST SENSOR PAIRS ==========")

    print(
        robust_pairs[
            [
                "failed_sensors",
                "MAE",
                "robust_degradation"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    print("\nDay 15 completed successfully.")

    print("Reports saved in:", REPORTS)


if __name__ == "__main__":
    main()
