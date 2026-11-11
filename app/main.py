from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

DECISIONS_FILE = REPORTS_DIR / "predictx_maintenance_decisions.csv"
PREDICTIONS_FILE = REPORTS_DIR / "nasa_test_predictions.csv"
SENSOR_HEALTH_FILE = REPORTS_DIR / "sensor_health_report.csv"
METRICS_FILE = REPORTS_DIR / "nasa_test_metrics.json"


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="PredictX AI API",
    description="Reliability-aware predictive maintenance API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")

    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_data():
    decisions = load_csv(DECISIONS_FILE)
    predictions = load_csv(PREDICTIONS_FILE)
    sensors = load_csv(SENSOR_HEALTH_FILE)

    required_columns = [
        "unit_id",
        "predicted_rul",
        "lower_rul",
        "upper_rul",
        "interval_width",
        "average_sensor_health",
        "unhealthy_sensor_count",
        "maintenance_priority",
        "recommended_action",
        "prediction_trust",
    ]

    missing = [
        column for column in required_columns
        if column not in decisions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in maintenance report: {missing}"
        )

    return decisions, predictions, sensors


def records_to_json(dataframe: pd.DataFrame):
    # Converts NumPy values and NaN into JSON-compatible values.
    return json.loads(dataframe.to_json(orient="records"))


def safe_mean(series):
    return round(float(pd.to_numeric(series, errors="coerce").mean()), 2)


# --------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "PredictX AI",
        "status": "running",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health():
    try:
        decisions, predictions, sensors = get_data()

        return {
            "status": "healthy",
            "backend": "online",
            "data": {
                "maintenance_records": len(decisions),
                "prediction_records": len(predictions),
                "sensor_records": len(sensors),
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )


# --------------------------------------------------
# DASHBOARD API
# --------------------------------------------------

@app.get("/api/dashboard")
def dashboard():
    try:
        decisions, predictions, sensors = get_data()

        # Operational engine records.
        # True RUL and prediction errors are deliberately excluded.
        engine_columns = [
            "unit_id",
            "last_cycle",
            "predicted_rul",
            "lower_rul",
            "upper_rul",
            "interval_width",
            "average_sensor_health",
            "unhealthy_sensor_count",
            "uncertainty_level",
            "sensor_health_status",
            "maintenance_priority",
            "recommended_action",
            "prediction_trust",
        ]

        engines = decisions[engine_columns].copy()
        engines = engines.sort_values("unit_id")

        # Dashboard KPIs.
        priority_counts = (
            decisions["maintenance_priority"]
            .value_counts()
            .to_dict()
        )

        summary = {
            "total_engines": int(len(decisions)),
            "high_priority": int(priority_counts.get("High", 0)),
            "medium_priority": int(priority_counts.get("Medium", 0)),
            "low_priority": int(priority_counts.get("Low", 0)),
            "average_rul": safe_mean(decisions["predicted_rul"]),
            "average_sensor_health": safe_mean(
                decisions["average_sensor_health"]
            ),
            "average_interval_width": safe_mean(
                decisions["interval_width"]
            ),
        }

        # Evaluation metrics are kept separate from operational predictions.
        evaluation_metrics = load_json(METRICS_FILE)

        return {
            "source": "NASA C-MAPSS FD001 reports",
            "summary": summary,
            "engines": records_to_json(engines),
            "evaluation": evaluation_metrics,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# --------------------------------------------------
# INDIVIDUAL ENGINE API
# --------------------------------------------------

@app.get("/api/engines/{unit_id}")
def engine_details(unit_id: int):
    try:
        decisions, predictions, sensors = get_data()

        engine_rows = decisions[
            decisions["unit_id"] == unit_id
        ]

        if engine_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Engine {unit_id} not found",
            )

        engine = engine_rows.iloc[0]

        # Sensor-level diagnostics for this engine.
        sensor_rows = sensors[
            sensors["unit_id"] == unit_id
        ].copy()

        sensor_rows = sensor_rows.sort_values("sensor")

        sensor_details = records_to_json(sensor_rows)

        # Operational response intentionally excludes true_rul.
        operational_fields = [
            "unit_id",
            "last_cycle",
            "predicted_rul",
            "lower_rul",
            "upper_rul",
            "interval_width",
            "covered",
            "average_sensor_health",
            "unhealthy_sensor_count",
            "uncertainty_level",
            "sensor_health_status",
            "maintenance_priority",
            "recommended_action",
            "prediction_trust",
        ]

        operational_data = {
            field: engine[field]
            for field in operational_fields
            if field in engine.index
        }

        return {
            "engine": json.loads(
                pd.Series(operational_data).to_json()
            ),
            "sensors": sensor_details,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
# SHAP EXPLAINABILITY API

@app.get("/api/explainability")
def explainability():
    try:
        global_data = load_csv(
            REPORTS_DIR / "shap_feature_importance.csv"
        )
        local_data = load_csv(
            REPORTS_DIR / "shap_local_explanation.csv"
        )

        global_data = global_data.sort_values(
            "mean_absolute_shap", ascending=False
        )

        local_data = local_data.sort_values(
            "absolute_shap", ascending=False
        )

        return {
            "global_importance": records_to_json(
                global_data.head(15)
            ),
            "local_explanation": records_to_json(
                local_data.head(15)
            ),
            "local_scope": (
                "Saved SHAP explanation. No engine ID is available."
            ),
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

