
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "model_train.csv"
VALIDATION_FILE = DATA_DIR / "model_validation.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.json"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_datasets():

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        FEATURE_FILE
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    with open(FEATURE_FILE, "r") as file:
        feature_columns = json.load(file)

    target_column = "RUL"

    if target_column not in train_df.columns:
        raise ValueError("RUL target missing from training data.")

    if target_column not in validation_df.columns:
        raise ValueError("RUL target missing from validation data.")

    missing_train = set(feature_columns) - set(train_df.columns)
    missing_validation = set(feature_columns) - set(validation_df.columns)

    if missing_train or missing_validation:
        raise ValueError(
            f"Missing features. Train: {missing_train}, "
            f"Validation: {missing_validation}"
        )

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(X_validation.to_numpy(dtype=float)).all():
        raise ValueError("Validation features contain NaN or infinity.")

    print("Datasets loaded successfully.")
    print("Training data:", X_train.shape)
    print("Validation data:", X_validation.shape)
    print("Number of features:", len(feature_columns))

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns
    )


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

def build_models():

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_validation, predictions)
    )

    r2 = r2_score(
        y_validation,
        predictions
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# --------------------------------------------------
# 5. TRAIN AND COMPARE
# --------------------------------------------------

def train_and_compare():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns
    ) = load_datasets()

    models = build_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 50)
        print("Training:", model_name)
        print("=" * 50)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        print("MAE:", round(metrics["MAE"], 4))
        print("RMSE:", round(metrics["RMSE"], 4))
        print("R2:", round(metrics["R2"], 4))

        trained_models[model_name] = model

        results.append({
            "Model": model_name,
            **metrics
        })

    # --------------------------------------------------
    # 6. SAVE TRAINED MODELS
    # --------------------------------------------------

    model_filenames = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost_model.pkl"
    }

    for model_name, model in trained_models.items():

        output_path = MODEL_DIR / model_filenames[model_name]

        joblib.dump(model, output_path)

        print(f"Saved {model_name}: {output_path}")

    # --------------------------------------------------
    # 7. SAVE COMPARISON REPORT
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE",
        ascending=True
    ).reset_index(drop=True)

    comparison_path = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(
        comparison_path,
        index=False
    )

    # Lowest validation MAE is the selection criterion.
    best_model_name = results_df.iloc[0]["Model"]

    best_model_info = {
        "best_model": best_model_name,
        "selection_metric": "MAE",
        "validation_MAE": float(
            results_df.iloc[0]["MAE"]
        ),
        "validation_RMSE": float(
            results_df.iloc[0]["RMSE"]
        ),
        "validation_R2": float(
            results_df.iloc[0]["R2"]
        ),
        "feature_count": len(feature_columns),
        "validation_method": "Engine-wise holdout"
    }

    with open(
        REPORT_DIR / "best_model.json",
        "w"
    ) as file:
        json.dump(best_model_info, file, indent=4)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(results_df.to_string(index=False))

    print("\nBest model based on validation MAE:")
    print(best_model_name)

    print("\nTraining and evaluation completed.")


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

if __name__ == "__main__":
    train_and_compare()
