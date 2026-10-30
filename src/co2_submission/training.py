import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .model import CO2Forecaster
from .losses import objective, objective_components
from .diagnostics import metric_rows

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def device_auto():return 'cpu'  # Explicit --device required for accelerator execution

def predict(model,data,device):
    if len(data['x'])==0:raise ValueError('Cannot predict empty dataset')
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(data['x']),256):
            x=torch.as_tensor(data['x'][start:start+256],dtype=torch.float32,device=device)
            result.append(model(x).cpu().numpy())
    return np.concatenate(result)

def train(data,scales,epochs=10,seed=42,device='cpu',kind='noise',mode='combined',reconstructed_weight=.05,lr=1e-3,
          validation=None,record_every=1,checkpoint_epochs=None,l1_proxy_weight=0.0,
          patience=None,min_delta=0.0,selection_metric="mae"):
    seed_all(seed)
    # Target calibration computed from genuinely observed TRAINING targets only.
    # No evaluation/test target enters this calculation.
    target_mean=[];target_scale=[]
    for j in range(6):
        values=data['y'][data['observed'][:,j].astype(bool),j]
        if len(values)==0:
            # L1 is not supervised; its output must not be interpreted as validated.
            values=data['y'][data['valid'][:,j].astype(bool),j]
        target_mean.append(float(np.mean(values)) if len(values) else 1.)
        target_scale.append(max(float(np.std(values)) if len(values)>1 else 1.,.01))
    model=CO2Forecaster(mode=mode,target_mean=target_mean,target_scale=target_scale).to(device)
    keys=('x','y','valid','observed');ds=TensorDataset(*(torch.as_tensor(data[k],dtype=torch.float32) for k in keys))
    dl=DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    s=torch.as_tensor(scales[kind],dtype=torch.float32,device=device)
    if patience is not None and (validation is None or record_every != 1):
        raise ValueError('Early stopping requires validation and per-epoch recording')
    if patience is not None and patience < 1:raise ValueError('patience must be >= 1')
    if min_delta < 0:raise ValueError('min_delta must be >= 0')
    if selection_metric not in ('mae','rmse'):raise ValueError('selection_metric must be mae or rmse')
    curve=[];locations=[];checkpoints={}
    best_score=float('inf');best_epoch=None;wait=0;stop_reason='max_epochs'
    best_state=None

    for ep in range(1,epochs+1):
        model.train();batch_losses=[];grad_norms=[];obs_contributions=[];rec_contributions=[];l1_contributions=[];active_location_counts=[]
        batch_target_weights=np.zeros(5,dtype=float);batch_target_counts=np.zeros(5,dtype=int);batch_observed_counts=np.zeros(5,dtype=int);batch_reconstructed_counts=np.zeros(5,dtype=int)
        for x,y,valid,obs in dl:
            x,y,valid,obs=[v.to(device) for v in (x,y,valid,obs)]
            # Record actual minibatch target exposure, not merely dataset-level counts.
            w=torch.where(obs[:,1:].bool(),torch.ones_like(y[:,1:]),torch.where(valid[:,1:].bool(),torch.full_like(y[:,1:],reconstructed_weight),torch.zeros_like(y[:,1:])))
            batch_target_weights+=w.sum(0).cpu().numpy()
            batch_target_counts+=(w.sum(0)>0).cpu().numpy().astype(int)
            batch_observed_counts+=obs[:,1:].sum(0).cpu().numpy().astype(int)
            batch_reconstructed_counts+=(valid[:,1:].bool() & ~obs[:,1:].bool()).sum(0).cpu().numpy().astype(int)
            opt.zero_grad(set_to_none=True)
            loss, parts=objective_components(model(x),y,valid.bool(),obs.bool(),s,kind,reconstructed_weight,l1_proxy_weight=l1_proxy_weight)
            if not parts['has_supervision']:
                continue  # No backward/AdamW step, including at reconstruction weight zero.
            obs_contributions.append(float(parts['observed'].detach().item()))
            rec_contributions.append(float(parts['reconstructed'].detach().item()))
            l1_contributions.append(float(parts['l1_proxy'].detach().item()))
            active_location_counts.append(parts['active_locations'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            batch_losses.append(float(loss.item()));grad_norms.append(float(gn.item()))
        if not batch_losses:
            raise RuntimeError('Training epoch has no supervised minibatches')
        if ep%record_every==0 or ep==epochs or ep==1:
            tr,loc=metric_rows(predict(model,data,device),data,scales,'train',ep,kind,reconstructed_weight,l1_proxy_weight)
            tr.update(optimizer_batch_loss=float(np.mean(batch_losses)),grad_norm=float(np.mean(grad_norms)),learning_rate=lr,
                      l1_proxy_loss_contribution=float(np.mean(l1_contributions)),
                      observed_loss_contribution=float(np.mean(obs_contributions)),
                      reconstructed_loss_contribution=float(np.mean(rec_contributions)),
                      observed_loss_fraction=float(np.sum(obs_contributions)/max(np.sum(obs_contributions)+np.sum(rec_contributions),1e-12)),
                      reconstructed_loss_fraction=float(np.sum(rec_contributions)/max(np.sum(obs_contributions)+np.sum(rec_contributions),1e-12)),
                      mean_active_locations=float(np.mean(active_location_counts)),
                      observed_target_count=int(batch_observed_counts.sum()),reconstructed_target_count=int(batch_reconstructed_counts.sum()))
            curve.append(tr);locations.extend(loc)
            for j in range(5):
                locations.append(dict(epoch=ep,split='train_exposure',active_objective=kind,
                                      evaluated_objective=kind,location=f'L{j+2}',
                                      n=int(batch_observed_counts[j]),reconstructed_n=int(batch_reconstructed_counts[j]),
                                      batches_with_targets=int(batch_target_counts[j]),
                                      effective_target_weight=float(batch_target_weights[j])))
            if validation is not None:
                va,vl=metric_rows(predict(model,validation,device),validation,scales,'inner_validation',ep,kind,reconstructed_weight,l1_proxy_weight)
                curve.append(va);locations.extend(vl)
                if patience is not None:
                    # Macro over observed L2-L6 only; never use reconstructed targets
                    # or the outer held-out experiment for stopping.
                    values=[r[selection_metric] for r in vl if r['evaluated_objective']==kind and r['location']!='L1' and r['n']>0 and np.isfinite(r[selection_metric])]
                    if not values:raise ValueError('No observed validation targets for early stopping')
                    score=float(np.mean(values))
                    improved=score < best_score - min_delta
                    if improved:
                        best_score=score;best_epoch=ep;wait=0
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    else:wait+=1
                    va.update(selection_score=score,best_selection_score=best_score,best_epoch=best_epoch,
                              patience_counter=wait,improved=improved)
                    tr.update(selection_score=np.nan,best_selection_score=best_score,best_epoch=best_epoch,
                              patience_counter=wait,improved=False)
        if checkpoint_epochs and ep in checkpoint_epochs:
            checkpoints[ep]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if patience is not None and wait >= patience:
            stop_reason='patience';break
    if patience is not None:
        if best_state is None:raise RuntimeError('No valid early-stopping checkpoint')
        # Return best model, not final (possibly overfit) epoch.
        model.load_state_dict(best_state)
        checkpoints['early_stopping']={'best_epoch':best_epoch,'best_score':best_score,
                                      'stopped_epoch':ep,'stop_reason':stop_reason,
                                      'state_dict':best_state}
    return model,curve,locations,checkpoints
