"""Checkpoint-faithful six-location inference. No refitting at serving time."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from co2_submission.features import FeatureSelector, FEATURES
from co2_submission.context import context
from co2_submission.model import CO2Forecaster

class ForecasterService:
    def __init__(self, checkpoint_path):
        self.path = Path(checkpoint_path)
        try:
            artifact = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Unable to load checkpoint {self.path}: {exc}") from exc
        for key in ("state_dict", "config", "features"):
            if key not in artifact:
                raise ValueError(f"Checkpoint missing {key}")
        self.config = artifact["config"]
        self.history = int(self.config["history"])
        self.horizon = int(self.config["horizon"])
        self.mode = self.config["mode"]
        if self.history < 1 or self.horizon < 1:
            raise ValueError("Invalid history/horizon")
        meta = artifact["features"]
        cols = meta["selected_features"]
        if list(cols) != list(FEATURES):
            raise ValueError("Checkpoint feature order differs from implementation")
        self.selector = FeatureSelector()
        self.selector.columns = list(cols)
        self.selector.median = np.asarray(meta["train_median"], dtype=float)
        self.selector.mean = np.asarray(meta["train_mean"], dtype=float)
        self.selector.std = np.asarray(meta["train_std"], dtype=float)
        if any(len(a) != 30 for a in (self.selector.median,self.selector.mean,self.selector.std)) or np.any(self.selector.std <= 0):
            raise ValueError("Invalid fitted preprocessing statistics")
        self.model = CO2Forecaster(mode=self.mode)
        self.model.load_state_dict(artifact["state_dict"], strict=True)
        self.model.eval()

    def predict_at_row(self, experiment: pd.DataFrame, origin: int):
        """Origin is a zero-based row index. No future rows are passed to feature builders."""
        if not isinstance(origin, int) or origin < self.history - 1 or origin >= len(experiment):
            raise ValueError(f"origin must be in [{self.history - 1}, {len(experiment)-1}]")
        historical = experiment.iloc[:origin + 1].copy().reset_index(drop=True)
        process = self.selector.transform(historical)
        co2, *_ = context(historical)
        x = np.concatenate([process, co2], axis=1)[-self.history:]
        if x.shape != (self.history, 54) or not np.isfinite(x).all():
            raise ValueError("Invalid inference features")
        with torch.inference_mode():
            y = self.model(torch.from_numpy(x[None].astype(np.float32))).cpu().numpy()[0]
        if y.shape != (6,) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid model output")
        return {"origin_row": origin,
                "forecast_origin": pd.Timestamp(historical.timestamp.iloc[-1]).isoformat(),
                "horizon_steps": self.horizon,
                "target_row": origin + self.horizon,
                "predictions": {f"L{i+1}": float(y[i]) for i in range(6)},
                "note": "Future timestamp not inferred from irregular sampling; target_row is the forecast index."}
