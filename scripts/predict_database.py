#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:53:40 2026

@author: niharikadeshpande
"""

"""Run the trained Transformer using PostgreSQL measurements."""

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from co2_submission.deployment.inference import ForecasterService

from co2_submission.deployment.repository import (
    load_experiment_from_database,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="results/final_test/final_model.pt",
    )

    parser.add_argument(
        "--experiment",
        default="140207_1.xlsx",
    )

    parser.add_argument(
        "--origin-row",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    service = ForecasterService(args.checkpoint)

    frame = load_experiment_from_database(
        args.experiment,
        origin_row=args.origin_row,
    )

    result = service.predict_at_row(
        frame,
        origin=args.origin_row,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()