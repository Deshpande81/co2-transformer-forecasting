"""Observed-only and weighted diagnostic metrics; never modify model or statistics."""
import numpy as np
import torch
from .losses import objective

KINDS=('raw','variance','noise')

def metric_rows(pred,data,scales,split,epoch,active,reconstructed_weight=.05,l1_proxy_weight=0.0):
    y=np.asarray(data['y']); obs=np.asarray(data['observed'],bool); valid=np.asarray(data['valid'],bool)
    p=torch.as_tensor(pred,dtype=torch.float32); yt=torch.as_tensor(y,dtype=torch.float32)
    vt=torch.as_tensor(valid); ot=torch.as_tensor(obs)
    result={'epoch':int(epoch),'split':split,'active_objective':active,'configured_reconstruction_weight':reconstructed_weight,'configured_l1_proxy_weight':l1_proxy_weight}
    for kind in KINDS:
        sc=torch.as_tensor(scales[kind],dtype=torch.float32)
        result[f'observed_{kind}_huber']=float(objective(p,yt,vt,ot,sc,kind,observed_only=True).item())
        if split=='train':
            result[f'full_{kind}_huber']=float(objective(p,yt,vt,ot,sc,kind,reconstructed_weight=reconstructed_weight,observed_only=False,l1_proxy_weight=l1_proxy_weight).item())
    l1mask=valid[:,0]
    result['l1_proxy_n']=int(l1mask.sum())
    result['l1_proxy_mae']=float(np.mean(np.abs((pred-y)[l1mask,0]))) if l1mask.any() else np.nan
    errs=(pred-y)[:,1:]; masks=obs[:,1:]
    e=errs[masks]
    result.update(observed_n=int(len(e)),observed_rmse=float(np.sqrt(np.mean(e*e))) if len(e) else np.nan,
                  observed_mae=float(np.mean(np.abs(e))) if len(e) else np.nan,
                  observed_bias=float(np.mean(e)) if len(e) else np.nan)
    locations=[]
    for j in range(1,6):
        m=obs[:,j]; d=pred[m,j]-y[m,j]
        for kind in KINDS:
            sc=float(scales[kind][j]);r=np.abs(d/sc)
            hub=np.where(r<=1,.5*r*r,r-.5)
            locations.append({'epoch':epoch,'split':split,'active_objective':active,'evaluated_objective':kind,
                              'location':f'L{j+1}','n':int(m.sum()),'huber':float(hub.mean()) if len(hub) else np.nan,
                              'rmse':float(np.sqrt(np.mean(d*d))) if len(d) else np.nan,
                              'mae':float(np.mean(np.abs(d))) if len(d) else np.nan,
                              'bias':float(np.mean(d)) if len(d) else np.nan})
    for kind in KINDS:
        group=[r for r in locations if r['evaluated_objective']==kind]
        denom=sum(r['huber'] for r in group if np.isfinite(r['huber']))
        for r in group:r['loss_share_pct']=100*r['huber']/denom if denom>0 and np.isfinite(r['huber']) else np.nan
    return result,locations

def prediction_rows(pred,data,split,config,fold,epoch):
    rows=[]
    for k in range(len(pred)):
        for j in range(6):
            rows.append({'split':split,'fold':fold,'experiment':str(data['run'][k]),'origin_row':int(data['origin'][k]),
                         'target_row':int(data['target'][k]),'location':f'L{j+1}',
                         'history':config['history'],'horizon':config['horizon'],
                         'input_mode':config['mode'],'training_objective':config['loss'],'epoch':epoch,
                         'y_true':float(data['y'][k,j]),'y_pred':float(pred[k,j]),
                         'observed':bool(data['observed'][k,j]),'valid':bool(data['valid'][k,j]),
                         'reconstructed':bool(data['valid'][k,j] and not data['observed'][k,j]),
                         'persistence_pred':float(data['baseline'][k,j]),
                         'persistence_available':bool(data['baseline_available'][k,j]),
                         'smoothed_pred':float(data['smoothed'][k,j]),
                         'smoothed_available':bool(data['smoothed_available'][k,j]),
                         'measurement_age_log_rows':float(data['measurement_age'][k,j])})
    return rows
