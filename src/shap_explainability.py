
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


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

MODEL_FILE = MODEL_DIR / "robust_random_forest.pkl"


# --------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------

BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 100
RANDOM_STATE = 42


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

def load_inputs():

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        FEATURE_FILE,
        MODEL_FILE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    features = pd.read_json(
        FEATURE_FILE,
        typ="series"
    ).tolist()

    model = joblib.load(MODEL_FILE)

    X_train = train_df[features].copy()
    X_validation = validation_df[features].copy()

    if not np.isfinite(
        X_train.to_numpy(dtype=float)
    ).all():
        raise ValueError("Training features contain invalid values.")

    if not np.isfinite(
        X_validation.to_numpy(dtype=float)
    ).all():
        raise ValueError("Validation features contain invalid values.")

    return (
        X_train,
        X_validation,
        validation_df,
        features,
        model
    )


# --------------------------------------------------
# 4. GENERATE SHAP EXPLANATIONS
# --------------------------------------------------

def calculate_shap_values(
    model,
    X_train,
    X_validation
):

    rng = np.random.default_rng(RANDOM_STATE)

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train)
    )

    explain_size = min(
        EXPLAIN_SIZE,
        len(X_validation)
    )

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False
    )

    explain_indices = rng.choice(
        len(X_validation),
        size=explain_size,
        replace=False
    )

    background = X_train.iloc[
        background_indices
    ]

    X_explain = X_validation.iloc[
        explain_indices
    ]

    print("\nBackground samples:", len(background))
    print("Samples to explain:", len(X_explain))

    # TreeExplainer is appropriate for tree-based models.
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_explain)

    return explainer, shap_values, X_explain


# --------------------------------------------------
# 5. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

def save_global_importance(
    shap_values,
    X_explain
):

    values = np.asarray(shap_values.values)

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP dimensions: {values.shape}"
        )

    mean_abs_shap = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_absolute_shap": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        by="mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nTop 10 influential features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


# --------------------------------------------------
# 6. GLOBAL SHAP VISUALIZATIONS
# --------------------------------------------------

def save_global_plots(shap_values):

    # Summary plot shows feature impact distribution.
    plt.figure(figsize=(12, 8))

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Bar plot shows average absolute SHAP importance.
    plt.figure(figsize=(12, 8))

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 7. LOCAL EXPLANATION
# --------------------------------------------------

def save_local_explanation(
    shap_values,
    X_explain
):

    # Explain the first selected validation observation.
    local_explanation = shap_values[0]

    local_df = pd.DataFrame({
        "feature": X_explain.columns,
        "feature_value": X_explain.iloc[0].values,
        "shap_value": local_explanation.values
    })

    local_df["absolute_shap"] = (
        local_df["shap_value"].abs()
    )

    local_df = local_df.sort_values(
        by="absolute_shap",
        ascending=False
    )

    local_df.to_csv(
        REPORT_DIR / "shap_local_explanation.csv",
        index=False
    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        local_explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nLocal explanation saved.")


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def run_explainability():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading model and datasets...")

    (
        X_train,
        X_validation,
        validation_df,
        features,
        model
    ) = load_inputs()

    print("Model loaded successfully.")
    print("Feature count:", len(features))

    (
        explainer,
        shap_values,
        X_explain
    ) = calculate_shap_values(
        model,
        X_train,
        X_validation
    )

    print("\nSHAP values calculated.")

    importance_df = save_global_importance(
        shap_values,
        X_explain
    )

    save_global_plots(shap_values)

    save_local_explanation(
        shap_values,
        X_explain
    )

    # Verify SHAP additive consistency.
    predictions = model.predict(X_explain)

    reconstructed = (
        np.asarray(shap_values.base_values).reshape(-1)
        + np.asarray(shap_values.values).sum(axis=1)
    )

    max_difference = float(
        np.max(np.abs(predictions - reconstructed))
    )

    print("\nMaximum SHAP reconstruction difference:")
    print(max_difference)

    print("\nTop five features:")
    print(
        importance_df.head(5).to_string(index=False)
    )

    print("\nAll SHAP reports saved successfully.")
    print("Day 11 explainability completed.")


if __name__ == "__main__":
    run_explainability()
