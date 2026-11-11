
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

BASELINE_MODEL = MODEL_DIR / "random_forest.pkl"
ROBUST_MODEL = MODEL_DIR / "robust_random_forest.pkl"


COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# --------------------------------------------------
# 2. LOAD TEST DATA
# --------------------------------------------------

def load_test_data():

    required_files = [
        TEST_FILE,
        RUL_FILE,
        FEATURE_FILE,
        BASELINE_MODEL,
        ROBUST_MODEL
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    test_df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None
    )

    test_df = test_df.dropna(axis=1, how="all")
    test_df.columns = COLUMNS

    true_rul = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    with open(FEATURE_FILE, "r") as file:
        features = json.load(file)

    baseline_model = joblib.load(BASELINE_MODEL)
    robust_model = joblib.load(ROBUST_MODEL)

    return (
        test_df,
        true_rul,
        features,
        baseline_model,
        robust_model
    )


# --------------------------------------------------
# 3. FEATURE ENGINEERING
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
# 4. PREPARE FINAL TEST ROWS
# --------------------------------------------------

def prepare_test_data(test_df, true_rul, features):

    engineered = create_features(
        test_df,
        window=5
    )

    final_rows = (
        engineered.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    if len(final_rows) != len(true_rul):
        raise ValueError(
            "Number of test engines and RUL labels do not match."
        )

    missing = [
        feature for feature in features
        if feature not in final_rows.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X_test = final_rows[features].copy()

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Invalid values found in test features.")

    y_test = true_rul.to_numpy()

    return final_rows, X_test, y_test


# --------------------------------------------------
# 5. EVALUATE BOTH MODELS
# --------------------------------------------------

def calculate_metrics(y_true, predictions, model_name):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# --------------------------------------------------
# 6. CREATE ERROR ANALYSIS
# --------------------------------------------------

def create_error_analysis(
    final_rows,
    y_test,
    baseline_predictions,
    robust_predictions
):

    results = pd.DataFrame({
        "unit_id": final_rows["unit_id"],
        "last_cycle": final_rows["cycle"],
        "true_rul": y_test,
        "baseline_prediction": baseline_predictions,
        "robust_prediction": robust_predictions
    })

    results["baseline_error"] = (
        results["baseline_prediction"] - results["true_rul"]
    )

    results["robust_error"] = (
        results["robust_prediction"] - results["true_rul"]
    )

    results["baseline_absolute_error"] = (
        results["baseline_error"].abs()
    )

    results["robust_absolute_error"] = (
        results["robust_error"].abs()
    )

    results["robust_improvement"] = (
        results["baseline_absolute_error"]
        - results["robust_absolute_error"]
    )

    results["robust_better"] = (
        results["robust_absolute_error"]
        < results["baseline_absolute_error"]
    )

    results.to_csv(
        REPORT_DIR / "test_error_analysis.csv",
        index=False
    )

    return results


# --------------------------------------------------
# 7. GENERATE PLOTS
# --------------------------------------------------

def generate_plots(results):

    # Actual vs predicted
    plt.figure(figsize=(9, 7))

    plt.scatter(
        results["true_rul"],
        results["baseline_prediction"],
        alpha=0.65,
        label="Baseline Random Forest"
    )

    plt.scatter(
        results["true_rul"],
        results["robust_prediction"],
        alpha=0.65,
        label="Robust Random Forest"
    )

    minimum = min(
        results["true_rul"].min(),
        results["baseline_prediction"].min(),
        results["robust_prediction"].min()
    )

    maximum = max(
        results["true_rul"].max(),
        results["baseline_prediction"].max(),
        results["robust_prediction"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs Predicted RUL")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "actual_vs_predicted.png",
        dpi=300
    )

    plt.close()

    # Residual comparison
    plt.figure(figsize=(9, 7))

    plt.scatter(
        results["true_rul"],
        results["baseline_error"],
        alpha=0.65,
        label="Baseline"
    )

    plt.scatter(
        results["true_rul"],
        results["robust_error"],
        alpha=0.65,
        label="Robust"
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Prediction Error (Predicted - Actual)")
    plt.title("Residual Analysis")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "residual_analysis.png",
        dpi=300
    )

    plt.close()

    # Absolute error distribution
    plt.figure(figsize=(9, 7))

    plt.hist(
        results["baseline_absolute_error"],
        bins=20,
        alpha=0.6,
        label="Baseline"
    )

    plt.hist(
        results["robust_absolute_error"],
        bins=20,
        alpha=0.6,
        label="Robust"
    )

    plt.xlabel("Absolute Prediction Error")
    plt.ylabel("Number of Engines")
    plt.title("Prediction Error Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "error_distribution.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading NASA test dataset and models...")

    (
        test_df,
        true_rul,
        features,
        baseline_model,
        robust_model
    ) = load_test_data()

    print("Test data loaded.")
    print("Test engines:", test_df["unit_id"].nunique())
    print("Features:", len(features))

    final_rows, X_test, y_test = prepare_test_data(
        test_df,
        true_rul,
        features
    )

    print("\nGenerating predictions...")

    baseline_predictions = baseline_model.predict(X_test)
    robust_predictions = robust_model.predict(X_test)

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
        "Baseline Random Forest"
    )

    robust_metrics = calculate_metrics(
        y_test,
        robust_predictions,
        "Robust Random Forest"
    )

    comparison = pd.DataFrame([
        baseline_metrics,
        robust_metrics
    ])

    comparison.to_csv(
        REPORT_DIR / "test_model_comparison.csv",
        index=False
    )

    results = create_error_analysis(
        final_rows,
        y_test,
        baseline_predictions,
        robust_predictions
    )

    generate_plots(results)

    robust_mae_change = (
        baseline_metrics["MAE"] - robust_metrics["MAE"]
    )

    print("\n========== TEST MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    print("\n========== ERROR ANALYSIS ==========")

    print(
        "Engines where robust model performed better:",
        int(results["robust_better"].sum())
    )

    print(
        "Engines where robust model did not perform better:",
        int((~results["robust_better"]).sum())
    )

    print(
        "Mean robust absolute error:",
        round(results["robust_absolute_error"].mean(), 4)
    )

    print(
        "Mean baseline absolute error:",
        round(results["baseline_absolute_error"].mean(), 4)
    )

    print(
        "MAE improvement (baseline - robust):",
        round(robust_mae_change, 4)
    )

    print("\nWorst 10 robust model predictions:")

    worst = results.nlargest(
        10,
        "robust_absolute_error"
    )

    print(
        worst[
            [
                "unit_id",
                "true_rul",
                "robust_prediction",
                "robust_absolute_error"
            ]
        ].to_string(index=False)
    )

    print("\nReports generated successfully.")


if __name__ == "__main__":
    main()
