import numpy as np
import torch
from co2_submission.losses import objective,objective_components
from co2_submission.training import train


def test_observed_only_unchanged_by_reconstruction_availability():
    pred=torch.zeros(4,6);y=torch.ones(4,6);valid=torch.ones(4,6,dtype=torch.bool)
    observed=torch.zeros_like(valid);observed[0]=True
    s=torch.ones(6)
    # Only L2-L6 contribute. Each observed residual has Huber 0.5.
    assert torch.isclose(objective(pred,y,valid,observed,s,reconstructed_weight=0),torch.tensor(.5))
    only_observed=observed.clone()
    assert torch.isclose(objective(pred,y,only_observed,observed,s,reconstructed_weight=0),torch.tensor(.5))


def test_components_and_weights():
    pred=torch.zeros(4,6);y=torch.ones(4,6);valid=torch.ones(4,6,dtype=torch.bool)
    observed=torch.zeros_like(valid);observed[0]=True
    s=torch.ones(6)
    for weight,expected in [(0,.5),(.05,.525),(1.,1.)]:
        value,parts=objective_components(pred,y,valid,observed,s,reconstructed_weight=weight)
        assert torch.isclose(value,torch.tensor(expected),atol=1e-6)
        assert torch.isclose(value,parts['observed']+parts['reconstructed'])


def test_reconstructed_only_no_supervision_at_zero_weight():
    pred=torch.zeros(4,6,requires_grad=True);y=torch.ones(4,6)
    valid=torch.ones(4,6,dtype=torch.bool);obs=torch.zeros_like(valid)
    val,parts=objective_components(pred,y,valid,obs,torch.ones(6),reconstructed_weight=0)
    assert val.item()==0 and not parts['has_supervision']
    val,parts=objective_components(pred,y,valid,obs,torch.ones(6),reconstructed_weight=.05)
    assert parts['has_supervision'] and torch.isclose(val,torch.tensor(.025))


def test_zero_weight_training_skips_unobserved_batches():
    rng=np.random.default_rng(42)
    n=32
    d={'x':rng.normal(size=(n,4,60)).astype('float32'),
       'y':np.ones((n,6),dtype='float32'),
       'valid':np.ones((n,6),dtype=bool),
       'observed':np.zeros((n,6),dtype=bool)}
    d['observed'][0,1]=True
    scales={k:np.ones(6,dtype='float32') for k in ('raw','variance','noise')}
    model,curve,loc,_=train(d,scales,epochs=1,kind='raw',mode='process',reconstructed_weight=0)
    assert len(curve)>0


def test_direct_model_learns_process_dependent_temporal_signal():
    from co2_submission.model import CO2Forecaster
    torch.manual_seed(7)
    n=96
    x=torch.randn(n,5,60)
    # Ground truth depends on the current and previous process measurements,
    # not on historical CO2 columns or a persistence anchor.
    signal=1.5+0.6*x[:,-1,0]+0.35*x[:,-2,1]
    y=torch.nn.functional.softplus(signal)
    model=CO2Forecaster(mode='process',width=16,layers=1,dropout=0,
        target_mean=[1.]*6,target_scale=[1.]*6)
    opt=torch.optim.Adam(model.parameters(),lr=.008)
    for _ in range(90):
        model.train();opt.zero_grad()
        loss=((model(x[:72, :])[:,1]-y[:72])**2).mean()
        loss.backward();opt.step()
    model.eval()
    with torch.no_grad():
        prediction=model(x[72:])[:,1]
        mse=((prediction-y[72:])**2).mean().item()
        constant=((y[72:]-y[:72].mean())**2).mean().item()
    assert mse < constant*.85, (mse,constant)
