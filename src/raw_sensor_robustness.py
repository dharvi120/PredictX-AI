
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. PATHS AND CONFIGURATION
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
MODEL_TRAIN_FILE = PROCESSED / "model_train.csv"

BASELINE_FILE = MODELS / "random_forest.pkl"
ROBUST_FILE = MODELS / "robust_random_forest.pkl"

SEEDS = [42, 21, 7, 100, 2026]
WINDOW = 5

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLUMNS = [
    f"sensor_{i}" for i in range(1, 22)
]


# ==================================================
# 2. LOAD DATA AND MODELS
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
            f"Expected 26 columns in {path.name}, "
            f"found {df.shape[1]}"
        )

    df.columns = COLUMNS

    return df


def load_inputs():

    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        RUL_FILE,
        FEATURE_FILE,
        MODEL_TRAIN_FILE,
        BASELINE_FILE,
        ROBUST_FILE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    train_raw = read_nasa_file(TRAIN_FILE)
    test_raw = read_nasa_file(TEST_FILE)

    true_rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    ).iloc[:, 0].to_numpy()

    with open(FEATURE_FILE, "r") as f:
        features = json.load(f)

    model_train = pd.read_csv(MODEL_TRAIN_FILE)

    baseline = joblib.load(BASELINE_FILE)
    robust = joblib.load(ROBUST_FILE)

    if len(test_raw["unit_id"].unique()) != len(true_rul):
        raise ValueError("Test engine and RUL label counts differ.")

    # Statistics come from training data only.
    raw_medians = train_raw[SENSOR_COLUMNS].median()
    raw_stds = train_raw[SENSOR_COLUMNS].std()

    # Fallback imputation values for engineered features.
    feature_medians = model_train[features].median()

    return (
        test_raw,
        true_rul,
        features,
        baseline,
        robust,
        raw_medians,
        raw_stds,
        feature_medians
    )


# ==================================================
# 3. FEATURE ENGINEERING
# Same logic as the training notebook
# ==================================================

def create_features(data):

    df = data.copy()

    df = df.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    for sensor in SENSOR_COLUMNS:

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
# 4. RAW SENSOR CORRUPTION
# ==================================================

def corrupt_raw_data(
    original,
    scenario,
    seed,
    raw_medians,
    raw_stds
):

    rng = np.random.default_rng(seed)

    corrupted = original.copy(deep=True)

    if scenario == "Clean":
        return corrupted

    if scenario == "Missing_10_Percent":

        for sensor in SENSOR_COLUMNS:

            mask = rng.random(len(corrupted)) < 0.10

            corrupted.loc[mask, sensor] = np.nan

        # Impute using training-set raw sensor medians.
        corrupted[SENSOR_COLUMNS] = (
            corrupted[SENSOR_COLUMNS]
            .fillna(raw_medians)
        )

    elif scenario == "Noise_20_Percent":

        for sensor in SENSOR_COLUMNS:

            noise = rng.normal(
                loc=0,
                scale=0.20 * raw_stds[sensor],
                size=len(corrupted)
            )

            corrupted[sensor] = (
                corrupted[sensor] + noise
            )

    elif scenario == "Dropout_2_Sensors":

        dropped_sensors = rng.choice(
            SENSOR_COLUMNS,
            size=2,
            replace=False
        )

        for sensor in dropped_sensors:

            corrupted[sensor] = raw_medians[sensor]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return corrupted


# ==================================================
# 5. PREPARE FINAL ENGINE FEATURES
# ==================================================

def prepare_features(
    corrupted_data,
    features,
    feature_medians
):

    engineered = create_features(corrupted_data)

    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    missing_features = [
        f for f in features
        if f not in final_rows.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = final_rows[features].copy()

    # Use training-derived medians, never test-derived statistics.
    X = X.fillna(feature_medians)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Invalid values in final test features.")

    return final_rows, X


# ==================================================
# 6. EVALUATION
# ==================================================

def evaluate_model(y_true, predictions):

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
# 7. RUN EXPERIMENTS
# ==================================================

def run_experiments():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading training statistics, test data and models...")

    (
        test_raw,
        true_rul,
        features,
        baseline,
        robust,
        raw_medians,
        raw_stds,
        feature_medians
    ) = load_inputs()

    print("Test engines:", test_raw["unit_id"].nunique())
    print("Features:", len(features))
    print("Random seeds:", SEEDS)

    scenarios = [
        "Clean",
        "Missing_10_Percent",
        "Noise_20_Percent",
        "Dropout_2_Sensors"
    ]

    all_results = []

    for scenario in scenarios:

        print(f"\nRunning scenario: {scenario}")

        for seed in SEEDS:

            corrupted = corrupt_raw_data(
                original=test_raw,
                scenario=scenario,
                seed=seed,
                raw_medians=raw_medians,
                raw_stds=raw_stds
            )

            final_rows, X_test = prepare_features(
                corrupted,
                features,
                feature_medians
            )

            if not np.array_equal(
                final_rows["unit_id"].to_numpy(),
                np.arange(1, len(true_rul) + 1)
            ):
                raise ValueError(
                    "Unexpected engine order in test data."
                )

            baseline_predictions = baseline.predict(X_test)
            robust_predictions = robust.predict(X_test)

            baseline_metrics = evaluate_model(
                true_rul,
                baseline_predictions
            )

            robust_metrics = evaluate_model(
                true_rul,
                robust_predictions
            )

            all_results.append({
                "scenario": scenario,
                "seed": seed,
                "model": "Baseline Random Forest",
                **baseline_metrics
            })

            all_results.append({
                "scenario": scenario,
                "seed": seed,
                "model": "Robust Random Forest",
                **robust_metrics
            })

            print(
                f"Seed {seed} | "
                f"Baseline MAE: {baseline_metrics['MAE']:.3f} | "
                f"Robust MAE: {robust_metrics['MAE']:.3f}"
            )

    # ==================================================
    # 8. SAVE RAW EXPERIMENT RESULTS
    # ==================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORTS / "raw_sensor_experiment_results.csv",
        index=False
    )

    summary = (
        results_df.groupby(["scenario", "model"])
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std")
        )
        .reset_index()
    )

    summary.to_csv(
        REPORTS / "raw_sensor_experiment_summary.csv",
        index=False
    )

    # Paired comparison: same scenario and same seed.
    paired = results_df.pivot(
        index=["scenario", "seed"],
        columns="model",
        values="MAE"
    ).reset_index()

    paired["baseline_minus_robust_MAE"] = (
        paired["Baseline Random Forest"]
        - paired["Robust Random Forest"]
    )

    paired.to_csv(
        REPORTS / "raw_sensor_paired_comparison.csv",
        index=False
    )

    # ==================================================
    # 9. DISPLAY SUMMARY
    # ==================================================

    print("\n========== RAW SENSOR EXPERIMENT SUMMARY ==========")

    print(
        summary.round(4).to_string(index=False)
    )

    print("\n========== PAIRED MAE DIFFERENCE ==========")

    print(
        paired.groupby("scenario")[
            "baseline_minus_robust_MAE"
        ].agg(["mean", "std"]).round(4).to_string()
    )

    print("\nAll experiments completed.")

    print("\nReports saved:")
    print("raw_sensor_experiment_results.csv")
    print("raw_sensor_experiment_summary.csv")
    print("raw_sensor_paired_comparison.csv")


if __name__ == "__main__":
    run_experiments()
