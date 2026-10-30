import numpy as np
import pandas as pd
import torch
from co2_submission.losses import objective
from co2_submission.features import FeatureSelector,FEATURES
from co2_submission.context import context
from co2_submission.reconstruction import interpolation_diagnostics

def test_missing_location_not_counted_in_batch_mean():
    p=torch.zeros(3,6,requires_grad=True)
    y=torch.zeros(3,6);y[:,1]=2.
    mask=torch.zeros(3,6,dtype=torch.bool);mask[:,1]=True
    loss=objective(p,y,mask,mask,torch.ones(6),kind='raw')
    assert torch.isclose(loss,torch.tensor(1.5))
    loss.backward();assert p.grad is not None

def test_causal_context_future_changes_do_not_change_past():
    d=pd.DataFrame({'label':[2,2,3,3,4,4,5,5], 'AT400':[1.,2.,3.,4.,5.,6.,7.,8.]})
    a=context(d)[0]; d.loc[6:,'AT400']=[900,1000]
    b=context(d)[0]
    np.testing.assert_array_equal(a[:6],b[:6])

def test_feature_scaler_training_only():
    def run(val):
        return pd.DataFrame({k:[val,val+1,val+2] for k in FEATURES})
    selector=FeatureSelector().fit({'training':run(1.)})
    before=selector.summary()
    selector.transform(run(10000.))
    assert before==selector.summary()

def test_interpolation_gaps():
    d=pd.DataFrame({'label':[2,2,3,3,2,2],'AT400':[1.,2.,3.,4.,5.,6.]})
    assert any(r['location']=='L2' and r['gap_rows']==2 for r in interpolation_diagnostics(d))
