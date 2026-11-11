
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

TEST_FILE = RAW_DIR / "test_FD001.txt"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)


def load_data():

    df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None
    )

    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS

    return df


def calculate_sensor_health(df):

    results = []

    for engine_id, engine_data in df.groupby("unit_id"):

        engine_data = engine_data.sort_values("cycle")

        for sensor in SENSORS:

            values = engine_data[sensor].astype(float)

            valid = values.dropna()

            if len(valid) == 0:
                continue

            missing_rate = float(values.isna().mean())

            # Consecutive equal readings.
            differences = valid.diff()

            stuck_rate = float(
                (differences == 0).mean()
            )

            # Robust outlier rate using median and MAD.
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

            # First-difference variation relative to
            # the overall sensor variation.
            changes = valid.diff().dropna()

            if len(changes) > 1:
                signal_std = float(valid.std())
                change_std = float(changes.std())

                noise_ratio = (
                    change_std / signal_std
                    if signal_std > 0
                    else 0.0
                )
            else:
                noise_ratio = 0.0

            # Heuristic penalties, each capped at 1.
            noise_penalty = min(noise_ratio / 2.0, 1.0)

            quality_score = 100 * (
                1
                - 0.35 * missing_rate
                - 0.30 * stuck_rate
                - 0.20 * min(outlier_rate, 1.0)
                - 0.15 * noise_penalty
            )

            quality_score = float(
                np.clip(quality_score, 0, 100)
            )

            if quality_score >= 85:
                status = "Healthy"
            elif quality_score >= 65:
                status = "Warning"
            else:
                status = "Critical"

            results.append({
                "unit_id": int(engine_id),
                "sensor": sensor,
                "missing_rate": missing_rate,
                "stuck_rate": stuck_rate,
                "outlier_rate": outlier_rate,
                "noise_ratio": noise_ratio,
                "health_score": quality_score,
                "status": status
            })

    return pd.DataFrame(results)


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading NASA FD001 test data...")

    df = load_data()

    print("Engines:", df["unit_id"].nunique())
    print("Sensors:", len(SENSORS))

    print("\nCalculating sensor health indicators...")

    results = calculate_sensor_health(df)

    results.to_csv(
        REPORT_DIR / "sensor_health_report.csv",
        index=False
    )

    summary = (
        results.groupby("sensor")
        .agg(
            average_health=("health_score", "mean"),
            minimum_health=("health_score", "min"),
            warning_engines=(
                "status",
                lambda s: (s == "Warning").sum()
            ),
            critical_engines=(
                "status",
                lambda s: (s == "Critical").sum()
            )
        )
        .reset_index()
        .sort_values("average_health")
    )

    summary.to_csv(
        REPORT_DIR / "sensor_health_summary.csv",
        index=False
    )

    print("\n========== SENSOR HEALTH SUMMARY ==========")

    print(summary.head(10).round(3).to_string(index=False))

    print("\n========== STATUS COUNTS ==========")

    print(
        results["status"].value_counts().to_string()
    )

    print("\nReports saved:")
    print("sensor_health_report.csv")
    print("sensor_health_summary.csv")


if __name__ == "__main__":
    main()
