import numpy as np
import pandas as pd
import torch
from co2_submission.context import context
from co2_submission.model import CO2Forecaster
from co2_submission.features import FeatureSelector, FEATURES
from co2_submission.sequences import windows
from co2_submission.reconstruction import targets

def frame(n=35):
    d=pd.DataFrame({c:np.linspace(1,2,n) for c in FEATURES})
    d['label']=np.resize(np.array([1,1,2,2,3,3,4,4,5,5,6,6]),n)
    d['AT400']=np.linspace(1,9,n)
    return d

def test_co2_context_has_no_duplicate_event_columns():
    x,*_=context(frame())
    assert x.shape[1]==24
    assert np.any(x[:,6:12]!=0)

def test_all_model_modes_match_feature_dimensions():
    x=torch.randn(3,8,54)
    for mode in ('process','co2','combined'):
        m=CO2Forecaster(mode=mode)
        y=m(x)
        assert y.shape==(3,6) and torch.isfinite(y).all() and (y>=0).all()

def test_evaluation_rows_independent_of_reconstruction_weight():
    df=frame();selector=FeatureSelector().fit({'train':df})
    a=windows({'eval':df},selector,6,2,reconstructed_weight=0)
    b=windows({'eval':df},selector,6,2,reconstructed_weight=.05)
    np.testing.assert_array_equal(a['target'],b['target'])
    np.testing.assert_array_equal(a['origin'],b['origin'])

def test_l1_is_only_proxy_and_does_not_use_contaminated_analyzer():
    df=frame();a=targets(df)
    df.loc[df.label==1,'AT400']=999999
    b=targets(df)
    np.testing.assert_array_equal(a[0][:,0],b[0][:,0])
    assert not a[2][:,0].any()
