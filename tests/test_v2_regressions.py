import numpy as np
import pandas as pd
import pytest
from co2_submission.features import FeatureSelector, FEATURES
from co2_submission.sequences import windows
from co2_submission.reconstruction import targets
from co2_submission.loader import load_runs

def fake(n=55, shift=0):
    d={f:np.arange(n,dtype=float)*.02+shift for f in FEATURES}
    d['label']=np.tile([2,3,4,5,6],(n+4)//5)[:n]
    d['AT400']=np.arange(n,dtype=float)*.1+1
    return pd.DataFrame(d)

def test_reconstruction_only_target_rows_are_retained():
    df=fake()
    df.loc[18:25,'label']=1  # no genuine L2-L6 observation; interpolation remains valid
    selector=FeatureSelector().fit({'train':df})
    d=windows({'train':df},selector,history=4,horizon=2)
    assert len(d['target'])>0
    assert np.any(~d['target_observed_any'])
    assert np.all(d['target_reconstructed_any'] | d['target_observed_any'])
    assert np.all(d['origin']<d['target'])

def test_train_fitted_imputation_scaling_and_holdout_invariance():
    train=fake(); train.loc[4:9,'FT100']=np.nan
    held=fake(shift=100);held.loc[4:9,'FT100']=np.nan
    sel=FeatureSelector().fit({'train':train})
    before=(sel.mean.copy(),sel.std.copy())
    a=sel.transform(held)
    assert np.isfinite(a).all()
    assert np.array_equal(before[0],sel.mean)
    assert np.array_equal(before[1],sel.std)
    # Independent causal forward-fill: altering later observations cannot affect earlier rows.
    modified=held.copy();modified.loc[40:,'FT100']=1e9
    b=sel.transform(modified)
    np.testing.assert_array_equal(a[:40],b[:40])

def test_all_feasible_valid_rows_included():
    df=fake(); sel=FeatureSelector().fit({'train':df})
    hist=6;horizon=3
    d=windows({'train':df},sel,hist,horizon)
    _,valid,_,_=targets(df)
    expected=np.flatnonzero(valid[hist-1+horizon:,1:].any(axis=1))+(hist-1+horizon)
    np.testing.assert_array_equal(d['target'],expected)

def test_duplicate_experiment_ids_fail_before_overwrite(tmp_path,monkeypatch):
    for filename in ['140120_1.xlsx','140120_1(1).xlsx']:
        (tmp_path/filename).touch()
    monkeypatch.setattr(pd,'read_excel',lambda *a,**k: pd.DataFrame({'Time':['2020-01-01'],'label':[2],'AT400':[1.]}))
    with pytest.raises(ValueError,match='Duplicate experiment ID'):
        load_runs(tmp_path)
