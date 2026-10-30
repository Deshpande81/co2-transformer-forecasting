"""Run the actual trained model before introducing PostgreSQL or Docker."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from co2_submission.loader import load_runs
from co2_submission.deployment.inference import ForecasterService

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--experiment", default="140207_1.xlsx")
    p.add_argument("--origin-row", type=int, required=True)
    a = p.parse_args()
    service = ForecasterService(a.checkpoint)
    runs = load_runs(a.raw_dir)
    if a.experiment not in runs: raise ValueError(f"Experiment not found: {a.experiment}")
    print(json.dumps(service.predict_at_row(runs[a.experiment], a.origin_row), indent=2))

if __name__ == "__main__": main()
