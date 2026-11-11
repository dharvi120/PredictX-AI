
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSORS
)

SEED = 42


def load_data():

    df = pd.read_csv(
        RAW / "train_FD001.txt",
        sep=r"\s+",
        header=None
    )

    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS

    return df


def inject_fault(df, sensor, fault_type, seed=SEED):

    rng = np.random.default_rng(seed)

    corrupted = df.copy(deep=True)
    original = df[sensor].copy()

    if fault_type == "missing":

        mask = rng.random(len(df)) < 0.10
        corrupted.loc[mask, sensor] = np.nan

    elif fault_type == "stuck":

        corrupted[sensor] = original.median()

    elif fault_type == "outlier":

        mask = rng.random(len(df)) < 0.10

        corrupted.loc[mask, sensor] = (
            original.median()
            + 10 * original.std()
        )

    elif fault_type == "noise":

        noise = rng.normal(
            0,
            2 * original.std(),
            len(df)
        )

        corrupted[sensor] = original + noise

    return corrupted


def detect_faults(df, sensor):

    values = df[sensor].astype(float)

    missing_rate = values.isna().mean()

    valid = values.dropna()

    stuck_rate = (
        valid.diff().eq(0).mean()
        if len(valid) > 1 else 0
    )

    median = valid.median()

    mad = np.median(
        np.abs(valid.to_numpy() - median)
    )

    if mad > 0:
        robust_z = (
            0.6745 * (valid - median).abs() / mad
        )

        outlier_rate = (
            robust_z > 3.5
        ).mean()
    else:
        outlier_rate = 0.0

    noise_ratio = (
        valid.diff().std() / valid.std()
        if len(valid) > 2 and valid.std() > 0
        else 0.0
    )

    # Transparent fault-detection rules.
    detected = []

    if missing_rate >= 0.05:
        detected.append("missing")

    if stuck_rate >= 0.80:
        detected.append("stuck")

    if outlier_rate >= 0.05:
        detected.append("outlier")

    if noise_ratio >= 1.50:
        detected.append("noise")

    return {
        "missing_rate": float(missing_rate),
        "stuck_rate": float(stuck_rate),
        "outlier_rate": float(outlier_rate),
        "noise_ratio": float(noise_ratio),
        "detected_faults": detected
    }


def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading training data...")

    df = load_data()

    test_sensor = "sensor_2"

    scenarios = [
        "missing",
        "stuck",
        "outlier",
        "noise"
    ]

    results = []

    for fault in scenarios:

        print("Testing:", fault)

        corrupted = inject_fault(
            df,
            test_sensor,
            fault
        )

        detection = detect_faults(
            corrupted,
            test_sensor
        )

        detected = detection["detected_faults"]

        results.append({
            "sensor": test_sensor,
            "actual_fault": fault,
            "detected_faults": ", ".join(detected),
            "correctly_detected": fault in detected,
            **{
                k: v for k, v in detection.items()
                if k != "detected_faults"
            }
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORTS / "sensor_health_validation.csv",
        index=False
    )

    print("\n========== SENSOR HEALTH VALIDATION ==========")

    print(
        results_df.to_string(index=False)
    )

    print(
        "\nCorrect detections:",
        int(results_df["correctly_detected"].sum()),
        "out of",
        len(results_df)
    )

    print(
        "\nSaved: reports/sensor_health_validation.csv"
    )


if __name__ == "__main__":
    main()
