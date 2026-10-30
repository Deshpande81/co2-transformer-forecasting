#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:02:16 2026

@author: niharikadeshpande
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "results/final_test/final_predictions.csv"
CHECKPOINT = ROOT / "results/final_test/final_model.pt"

ORIGIN_ROWS = [20, 40, 60, 80, 100]

LOCATIONS = [f"L{i}" for i in range(1, 7)]

TOLERANCE = 1e-5


def test_final_inference_matches_offline_evaluation():

    reference = pd.read_csv(PREDICTIONS)

    for origin_row in ORIGIN_ROWS:

        command = [
            sys.executable,
            str(ROOT / "scripts/predict_local.py"),
            "--checkpoint",
            str(CHECKPOINT),
            "--raw-dir",
            str(ROOT / "data/raw"),
            "--experiment",
            "140207_1.xlsx",
            "--origin-row",
            str(origin_row),
        ]

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        deployment = json.loads(result.stdout)

        expected = reference[
            (reference["experiment"] == "140207_1.xlsx")
            & (reference["origin_row"] == origin_row)
            & (reference["horizon"] == 1)
        ]

        assert len(expected) == 6, (
            f"Expected six predictions at origin {origin_row}, "
            f"found {len(expected)}"
        )

        assert expected["location"].is_unique

        expected_by_location = expected.set_index("location")["y_pred"]

        differences = []

        for location in LOCATIONS:

            actual = float(deployment["predictions"][location])
            target = float(expected_by_location.loc[location])

            assert np.isfinite(actual)
            assert actual >= 0

            differences.append(abs(actual - target))

        max_difference = max(differences)

        print(
            f"Origin {origin_row}: "
            f"maximum absolute difference = {max_difference:.10f}"
        )

        assert max_difference <= TOLERANCE, (
            f"Inference mismatch at origin {origin_row}: "
            f"{max_difference}"
        )