import numpy as np
import pandas as pd
import pytest
import torch
from co2_submission.reconstruction import targets
from co2_submission.losses import objective_components
from co2_submission.diagnostics import metric_rows
from co2_submission.noise import scales_from_training
from co2_submission.features import FeatureSelector,FEATURES
from co2_submission.sequences import windows

def test_l1_proxy_uses_mean_of_following_l2_dwell_not_interpolated_l2():
    d=pd.DataFrame({'label':[1,1,2,2,3,3,1,1,2,2,3,3],
                    'AT400':[99.,99.,2.,4.,6.,7.,99.,99.,8.,10.,11.,12.]})
    y,v,o,p=targets(d)
    assert not o[:,0].any()
    assert p[:,0].any()
    np.testing.assert_allclose(y[0:2,0],3.)
    np.testing.assert_allclose(y[6:8,0],9.)
    assert not np.allclose(y[v[:,0],0],y[v[:,0],1])
    assert not v[-1,0]  # no extrapolation beyond final L1 anchor
    np.testing.assert_array_equal(p[:,0],v[:,0])


def test_diagnostic_weight_is_configured_and_matches_objective():
    y=np.ones((4,6),dtype='float32'); pred=np.zeros_like(y)
    valid=np.ones_like(y,dtype=bool);observed=np.zeros_like(valid);observed[0,1:]=True
    valid[:,0]=False
    data=dict(y=y,observed=observed,valid=valid)
    scales={k:np.ones(6,dtype='float32') for k in ('raw','variance','noise')}
    for weight in (0.,.05,1.):
        r,_=metric_rows(pred,data,scales,'train',1,'raw',reconstructed_weight=weight)
        expected,_=objective_components(torch.from_numpy(pred),torch.from_numpy(y),torch.from_numpy(valid),torch.from_numpy(observed),torch.ones(6),kind='raw',reconstructed_weight=weight)
        assert r['full_raw_huber']==pytest.approx(expected.item())
        assert r['configured_reconstruction_weight']==weight
    r,_=metric_rows(pred,data,scales,'train',1,'raw',reconstructed_weight=.05,l1_proxy_weight=.05)
    assert r['full_raw_huber']==pytest.approx(.525)

def test_l1_proxy_gradients_and_weight():
    p=torch.zeros(4,6,requires_grad=True);y=torch.ones_like(p)
    v=torch.zeros_like(p,dtype=torch.bool);v[:,0]=True;o=torch.zeros_like(v)
    for weight in (0.,.05,1.):
        loss,parts=objective_components(p,y,v,o,torch.ones(6),reconstructed_weight=0,l1_proxy_weight=weight)
        assert loss.item()==pytest.approx(weight*.5)
        assert parts['has_supervision']==(weight>0)
    loss.backward();assert p.grad[:,0].abs().sum()>0
    assert p.grad[:,1:].abs().sum()==0

def test_future_targets_never_enter_history_inputs():
    n=30
    df=pd.DataFrame({k:np.arange(n,dtype=float) for k in FEATURES})
    df['label']=np.tile([2,2,3,3,4,4],5);df['AT400']=np.arange(n,dtype=float)
    sel=FeatureSelector().fit({'a':df})
    a=windows({'a':df},sel,history=6,horizon=2)
    df2=df.copy();df2.loc[15:,'AT400']+=100000
    b=windows({'a':df2},sel,history=6,horizon=2)
    origin=a['origin']
    np.testing.assert_allclose(a['x'][origin<15],b['x'][origin<15])
    assert (a['target']==a['origin']+2).all()
