"""Compare deployed FastAPI forecasts with saved independent-test predictions."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


REFERENCE = Path("results/final_test/final_predictions.csv")
API_URL = "http://127.0.0.1:8000/predict"
OUTPUT = Path("results/deployment_validation")

TOLERANCE = 1e-5
EXPERIMENT = "140207_1.xlsx"


def get_api_prediction(origin_row):
    payload = json.dumps({
        "experiment": EXPERIMENT,
        "origin_row": int(origin_row),
    }).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REFERENCE)

    df = df[
        (df["split"] == "final_test")
        & (df["experiment"] == EXPERIMENT)
        & (df["input_mode"] == "combined")
        & (df["training_objective"] == "noise")
        & (df["history"] == 12)
        & (df["horizon"] == 1)
        & (df["epoch"] == 43)
    ].copy()

    assert not df.empty, "No matching offline predictions found."

    assert not df.duplicated(
        ["origin_row", "location"]
    ).any(), "Duplicate origin/location pairs."

    locations = {f"L{i}" for i in range(1, 7)}

    origins = sorted(df["origin_row"].unique())

    assert len(origins) == 106, (
        f"Expected 106 forecast origins; found {len(origins)}"
    )

    assert len(df) == len(origins) * 6, (
        "Incomplete offline prediction coverage."
    )

    records = []

    for origin in origins:
        reference = df[df["origin_row"] == origin]

        assert set(reference["location"]) == locations, (
            f"Missing locations at origin {origin}"
        )

        response = get_api_prediction(origin)

        assert response["origin_row"] == int(origin)
        assert response["horizon_steps"] == 1
        assert response["target_row"] == int(origin) + 1

        api_predictions = response["predictions"]

        assert set(api_predictions) == locations

        for row in reference.itertuples():
            offline_value = float(row.y_pred)
            api_value = float(api_predictions[row.location])

            records.append({
                "experiment": EXPERIMENT,
                "origin_row": int(origin),
                "target_row": int(row.target_row),
                "location": row.location,
                "offline_prediction": offline_value,
                "api_prediction": api_value,
                "absolute_difference": abs(
                    offline_value - api_value
                ),
            })

    comparison = pd.DataFrame(records)

    assert np.isfinite(
        comparison[
            ["offline_prediction", "api_prediction", "absolute_difference"]
        ].to_numpy()
    ).all(), "Non-finite predictions detected."

    max_difference = comparison["absolute_difference"].max()
    mean_difference = comparison["absolute_difference"].mean()

    comparison["passed"] = (
        comparison["absolute_difference"] <= TOLERANCE
    )

    comparison.to_csv(
        OUTPUT / "offline_api_comparison.csv",
        index=False,
    )

    summary = {
        "experiment": EXPERIMENT,
        "history": 12,
        "horizon": 1,
        "forecast_origins": len(origins),
        "predictions_compared": len(comparison),
        "tolerance": TOLERANCE,
        "maximum_absolute_difference": float(max_difference),
        "mean_absolute_difference": float(mean_difference),
        "failed_predictions": int((~comparison["passed"]).sum()),
        "status": (
            "PASS"
            if comparison["passed"].all()
            else "FAIL"
        ),
    }

    with open(
        OUTPUT / "validation_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nDEPLOYMENT CONSISTENCY VALIDATION")
    print("--------------------------------")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Forecast origins: {len(origins)}")
    print(f"Predictions compared: {len(comparison)}")
    print(f"Maximum absolute difference: {max_difference:.10e}")
    print(f"Mean absolute difference: {mean_difference:.10e}")
    print(f"Failed predictions: {summary['failed_predictions']}")
    print(f"\nRESULT: {summary['status']}")

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
