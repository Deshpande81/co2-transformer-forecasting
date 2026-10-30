#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:01:59 2026

@author: niharikadeshpande
"""

"""Initialize PostgreSQL and ingest the CO2 experimental datasets."""

import json
import sys
from pathlib import Path

from sqlalchemy import func, select

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from co2_submission.deployment.database import (
    create_tables,
    Experiment,
    ProcessMeasurement,
    CO2Observation,
    get_session_factory,
)

from co2_submission.deployment.ingestion import ingest_experiments


EXPECTED_ROWS = {
    "140120_1.xlsx": 117,
    "140206_1.xlsx": 50,
    "140207_1.xlsx": 118,
    "140207_2.xlsx": 50,
    "140214_1.xlsx": 67,
    "140214_2.xlsx": 50,
    "140227_1.xlsx": 218,
    "140313_1.xlsx": 228,
}


def verify_database():
    Session = get_session_factory()

    with Session() as session:
        experiments = session.execute(
            select(
                Experiment.name,
                func.count(ProcessMeasurement.id),
            )
            .outerjoin(
                ProcessMeasurement,
                ProcessMeasurement.experiment_id == Experiment.id,
            )
            .group_by(Experiment.name)
        ).all()

        process_counts = dict(experiments)

        observations = session.execute(
            select(
                Experiment.name,
                func.count(CO2Observation.id),
            )
            .outerjoin(
                CO2Observation,
                CO2Observation.experiment_id == Experiment.id,
            )
            .group_by(Experiment.name)
        ).all()

        observation_counts = dict(observations)

    if process_counts != EXPECTED_ROWS:
        raise RuntimeError(
            f"Unexpected process measurement counts: {process_counts}"
        )

    if observation_counts != EXPECTED_ROWS:
        raise RuntimeError(
            f"Unexpected CO2 observation counts: {observation_counts}"
        )

    print("Database verification passed.")
    print(f"Experiments: {len(process_counts)}")
    print(f"Process measurements: {sum(process_counts.values())}")
    print(f"CO2 observations: {sum(observation_counts.values())}")


def main():
    print("Creating database tables...", flush=True)
    create_tables()

    print("Ingesting experimental datasets...", flush=True)
    summary = ingest_experiments("/app/data/raw")

    print(json.dumps(summary, indent=2), flush=True)

    print("Verifying database contents...", flush=True)
    verify_database()

    print("DATABASE INITIALIZATION SUCCESSFUL", flush=True)


if __name__ == "__main__":
    main()