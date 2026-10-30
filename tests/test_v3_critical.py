import numpy as np
import torch
from co2_submission.losses import objective, objective_components


def test_reconstructed_only_is_actually_downweighted():
    p=torch.zeros(20,6); y=torch.ones(20,6)
    v=torch.ones(20,6,dtype=torch.bool)
    o=torch.zeros_like(v)
    s=torch.ones(6)
    a=objective(p,y,v,o,s,reconstructed_weight=.05)
    b=objective(p,y,v,o,s,reconstructed_weight=1.)
    assert torch.allclose(a/b,torch.tensor(.05),atol=1e-6)
    assert torch.allclose(objective(p,y,v,o,s,reconstructed_weight=0),torch.tensor(0.))


def test_observed_and_reconstructed_components_add_to_total():
    p=torch.zeros(4,6); y=torch.ones(4,6)
    v=torch.ones(4,6,dtype=torch.bool);o=torch.zeros_like(v)
    o[:2,:]=True
    total,c=objective_components(p,y,v,o,torch.ones(6),reconstructed_weight=.05)
    assert torch.allclose(total,c['observed']+c['reconstructed'])
    assert abs(c['reconstructed'].item()/c['observed'].item()-.05)<1e-6


def test_direct_transformer_can_fit_synthetic_constant_levels():
    from co2_submission.model import CO2Forecaster
    torch.manual_seed(42)
    x=torch.randn(24,5,60)
    # The mapping is independent of last CO2 measurements, proving direct output learning.
    target=torch.tensor([.1,.5,1.,2.,4.,6.])[None,:].expand(24,-1)
    model=CO2Forecaster(mode='process',width=16,layers=1,target_mean=[.2,.4,.8,1.5,3.,5.],target_scale=[1.]*6)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    model.train()
    with torch.no_grad(): initial=((model(x)-target)**2).mean().item()
    for _ in range(35):
        opt.zero_grad();loss=((model(x)-target)**2).mean();loss.backward();opt.step()
    model.eval()
    with torch.no_grad(): final=((model(x)-target)**2).mean().item()
    assert final < initial*.4, (initial,final)
    assert (model(x)>=0).all()
