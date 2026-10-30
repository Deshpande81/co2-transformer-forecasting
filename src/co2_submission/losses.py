"""Location-macro Huber loss with independently weighted observed/reconstructed means.

For each L2-L6 location: mean(observed losses) + lambda * mean(reconstructed losses).
If a component is absent, it contributes zero. A reconstructed-only location
contributes lambda times its reconstructed mean, without dilution of observations.
"""
import torch
import torch.nn.functional as F


def objective_components(pred, y, valid, observed, scale, kind='noise', reconstructed_weight=.05, observed_only=False, l1_proxy_weight=0.0):
    if kind not in ('raw','variance','noise'): raise ValueError(kind)
    if not 0 <= reconstructed_weight <= 1: raise ValueError('reconstructed_weight must be in [0,1]')
    if not 0 <= l1_proxy_weight <= 1: raise ValueError('l1_proxy_weight must be in [0,1]')
    valid=valid.bool(); observed=observed.bool() & valid
    reconstructed=valid & ~observed
    residual=(pred[:,1:]-y[:,1:])/scale[None,1:]
    per_target=F.huber_loss(residual,torch.zeros_like(residual),delta=1.,reduction='none')
    om=observed[:,1:].float()
    rm=(reconstructed[:,1:].float() if not observed_only and reconstructed_weight>0 else torch.zeros_like(om))
    no=om.sum(dim=0);nr=rm.sum(dim=0)
    obs_mean=(per_target*om).sum(dim=0)/no.clamp_min(1)
    rec_mean=(per_target*rm).sum(dim=0)/nr.clamp_min(1)
    active=(no>0)|(nr>0)
    l1_active=bool(valid[:,0].any()) and not observed_only and l1_proxy_weight>0
    if not bool(active.any()) and not l1_active:
        zero=pred.sum()*0
        return zero,dict(observed=zero,reconstructed=zero,active_locations=0,has_supervision=False,
                         observed_count=0,reconstructed_count=0,l1_proxy=zero,l1_proxy_count=0)
    obs_part=obs_mean[active].mean() if bool(active.any()) else pred.sum()*0
    rec_part=(reconstructed_weight*rec_mean[active]).mean() if bool(active.any()) else pred.sum()*0
    l1_mask=valid[:,0]
    l1_loss=F.huber_loss((pred[l1_mask,0]-y[l1_mask,0])/scale[0],torch.zeros_like(pred[l1_mask,0]),delta=1.,reduction='mean')*l1_proxy_weight if l1_active else pred.sum()*0
    return obs_part+rec_part+l1_loss,dict(observed=obs_part,reconstructed=rec_part,l1_proxy=l1_loss,
        active_locations=int(active.sum().item())+int(l1_active),has_supervision=True,
        observed_count=int(no.sum().item()),reconstructed_count=int(nr.sum().item()),l1_proxy_count=int(l1_mask.sum().item()) if l1_active else 0)


def objective(pred,y,valid,observed,scale,kind='noise',reconstructed_weight=.05,observed_only=False,l1_proxy_weight=0.0):
    return objective_components(pred,y,valid,observed,scale,kind,reconstructed_weight,observed_only,l1_proxy_weight)[0]
