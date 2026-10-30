import numpy as np,torch
from co2_submission.model import CO2Forecaster
from co2_submission.losses import objective
from co2_submission.context import context
import pandas as pd

def test_direct_positive():
    m=CO2Forecaster(mode='combined');z=m(torch.randn(2,18,54));assert z.shape==(2,6) and (z>=0).all()
def test_modes():
    for mode in ('process','co2','combined'):
        assert CO2Forecaster(mode=mode)(torch.randn(2,12,54)).shape==(2,6)
def test_losses():
    p=torch.ones(4,6,requires_grad=True);y=torch.zeros_like(p);m=torch.ones_like(p,dtype=torch.bool)
    for kind in ('raw','variance','noise'):
        loss=objective(p,y,m,m,torch.ones(6),kind);assert torch.isfinite(loss);loss.backward(retain_graph=True)
def test_dwell_no_future():
    d=pd.DataFrame({'label':[2,2,3,3,4,4],'AT400':[1.,2.,3.,4.,5.,6.]})
    a,*_=context(d);b=d.copy();b.loc[4:,'AT400']=999
    c,*_=context(b)
    assert np.array_equal(a[:4],c[:4]);assert a[1,1]==0 and a[2,1]==1.5
