
from pathlib import Path
import pandas as pd
import numpy as np
import json


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CONFORMAL_FILE = REPORTS / "conformal_coverage_results.csv"
PIPELINE_FILE = REPORTS / "predictx_pipeline_output.csv"

OUTPUT_FILE = REPORTS / "predictx_maintenance_decisions.csv"
SUMMARY_FILE = REPORTS / "predictx_decision_summary.json"


def main():

    print("Loading prediction and sensor health reports...")

    for file in [CONFORMAL_FILE, PIPELINE_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Required report not found: {file}"
            )

    predictions = pd.read_csv(CONFORMAL_FILE)
    health = pd.read_csv(PIPELINE_FILE)

    required_prediction_columns = [
        "unit_id",
        "predicted_rul",
        "lower_rul",
        "upper_rul",
        "interval_width"
    ]

    required_health_columns = [
        "unit_id",
        "average_sensor_health",
        "unhealthy_sensor_count"
    ]

    for col in required_prediction_columns:
        if col not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {col}"
            )

    for col in required_health_columns:
        if col not in health.columns:
            raise ValueError(
                f"Missing health column: {col}"
            )

    print("Merging engine-level reports...")

    health = health[
        [
            "unit_id",
            "average_sensor_health",
            "unhealthy_sensor_count"
        ]
    ].drop_duplicates(subset=["unit_id"])

    report = predictions.merge(
        health,
        on="unit_id",
        how="left",
        validate="one_to_one"
    )

    if report["average_sensor_health"].isna().any():
        raise ValueError(
            "Some engines are missing sensor health information."
        )

    # ------------------------------------------------
    # UNCERTAINTY CLASSIFICATION
    # ------------------------------------------------

    report["uncertainty_level"] = np.select(
        [
            report["interval_width"] > 150,
            report["interval_width"] > 100
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    # ------------------------------------------------
    # SENSOR HEALTH CLASSIFICATION
    # ------------------------------------------------

    report["sensor_health_status"] = np.select(
        [
            report["average_sensor_health"] < 65,
            report["average_sensor_health"] < 85
        ],
        [
            "Critical",
            "Warning"
        ],
        default="Healthy"
    )

    # ------------------------------------------------
    # TRANSPARENT DECISION RULES
    # ------------------------------------------------

    high_priority = (
        (report["predicted_rul"] <= 30)
        | (report["average_sensor_health"] < 65)
    )

    medium_priority = (
        (report["predicted_rul"] <= 60)
        | (report["average_sensor_health"] < 85)
        | (report["interval_width"] > 150)
    )

    report["maintenance_priority"] = np.select(
        [
            high_priority,
            medium_priority
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    # ------------------------------------------------
    # RECOMMENDED ACTION
    # ------------------------------------------------

    report["recommended_action"] = np.select(
        [
            report["maintenance_priority"] == "High",
            report["maintenance_priority"] == "Medium"
        ],
        [
            "Prioritise inspection and verify sensor readings",
            "Schedule inspection and monitor RUL trend"
        ],
        default="Continue routine monitoring"
    )

    # ------------------------------------------------
    # PREDICTION TRUST FLAG
    # ------------------------------------------------

    report["prediction_trust"] = np.select(
        [
            (
                (report["average_sensor_health"] < 65)
                | (report["interval_width"] > 150)
            ),
            (
                (report["average_sensor_health"] < 85)
                | (report["interval_width"] > 100)
            )
        ],
        [
            "Use with caution",
            "Review before action"
        ],
        default="Normal review"
    )

    report = report.sort_values(
        ["maintenance_priority", "predicted_rul"],
        ascending=[True, True]
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary = {
        "total_engines": int(len(report)),
        "priority_counts": {
            str(key): int(value)
            for key, value in
            report["maintenance_priority"].value_counts().items()
        },
        "sensor_health_status_counts": {
            str(key): int(value)
            for key, value in
            report["sensor_health_status"].value_counts().items()
        },
        "uncertainty_level_counts": {
            str(key): int(value)
            for key, value in
            report["uncertainty_level"].value_counts().items()
        },
        "average_predicted_rul": float(
            report["predicted_rul"].mean()
        ),
        "average_sensor_health": float(
            report["average_sensor_health"].mean()
        ),
        "average_interval_width": float(
            report["interval_width"].mean()
        ),
        "decision_rules": {
            "high": "Predicted RUL <= 30 OR sensor health < 65",
            "medium": "Predicted RUL <= 60 OR sensor health < 85 OR interval width > 150",
            "low": "None of the above conditions"
        }
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n========== PREDICTX DECISION SUMMARY ==========")
    print(json.dumps(summary, indent=4))

    print("\nFirst ten maintenance decisions:")
    print(
        report[
            [
                "unit_id",
                "predicted_rul",
                "lower_rul",
                "upper_rul",
                "average_sensor_health",
                "maintenance_priority",
                "prediction_trust",
                "recommended_action"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()
