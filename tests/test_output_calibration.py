import numpy as np
import torch
from co2_submission.model import CO2Forecaster

def test_initial_output_matches_training_mean():
    model=CO2Forecaster(target_mean=[.01,.02,.1,.4,2.,5.],target_scale=[.01,.01,.1,.2,.5,1.])
    with torch.no_grad():
        y=model(torch.randn(3,18,54)).numpy()
    assert np.allclose(y,[.01,.02,.1,.4,2.,5.],atol=1e-5)
    assert np.all(y>=0)

def test_predictions_not_algebraically_anchored_to_co2_input():
    model=CO2Forecaster(mode='combined',target_mean=[.01,.02,.1,.4,2.,5.])
    with torch.no_grad():
        a=model(torch.zeros(2,18,54));b=model(torch.ones(2,18,54))
    # At initialization the output is train-only calibrated, not last-dwell value.
    assert torch.allclose(a,b)
