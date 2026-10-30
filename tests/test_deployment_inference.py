import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from co2_submission.features import FEATURES
from co2_submission.model import CO2Forecaster
from co2_submission.context import context
from co2_submission.deployment.inference import ForecasterService

def sample_data(n=24):
    data={c: np.arange(n, dtype=float) for c in FEATURES}
    data.update(timestamp=pd.date_range("2020-01-01", periods=n, freq="43s"),
                label=[2]*6+[3]*6+[4]*6+[5]*(n-18), AT400=np.linspace(0.2,1.2,n))
    return pd.DataFrame(data)

def test_checkpoint_and_prefix_causality(tmp_path):
    model=CO2Forecaster(mode="combined")
    checkpoint=tmp_path/"final_model.pt"
    torch.save({"state_dict":model.state_dict(),"config":{"history":12,"horizon":1,"mode":"combined"},
                "features":{"selected_features":list(FEATURES),"train_median":[0.]*30,
                            "train_mean":[0.]*30,"train_std":[1.]*30}},checkpoint)
    svc=ForecasterService(checkpoint)
    df=sample_data()
    first=svc.predict_at_row(df, 13)
    assert len(first["predictions"])==6
    assert all(v>=0 for v in first["predictions"].values())
    changed=df.copy()
    changed.loc[14:, "AT400"]=9999
    changed.loc[14:, "FT100"]=9999
    assert svc.predict_at_row(changed,13)["predictions"]==first["predictions"]
    with pytest.raises(ValueError):svc.predict_at_row(df, 5)
