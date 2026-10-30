#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:50:38 2026

@author: niharikadeshpande
"""

"""Populate PostgreSQL with experimental measurements."""

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from co2_submission.deployment.ingestion import ingest_experiments


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
    )

    args = parser.parse_args()

    summary = ingest_experiments(args.raw_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()