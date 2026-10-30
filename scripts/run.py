"""Leakage-separated nested LOEO, epoch curves, and complete prediction exports."""
import argparse,json,sys,time,platform
from pathlib import Path
from itertools import product
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from co2_submission.loader import load_runs
from co2_submission.features import FeatureSelector
from co2_submission.sequences import windows
from co2_submission.noise import scales_from_training
from co2_submission.training import train,predict,device_auto
from co2_submission.diagnostics import prediction_rows
from co2_submission.model import CO2Forecaster
from co2_submission.reconstruction import interpolation_diagnostics

def save(rows,path):
    if rows:pd.DataFrame(rows).to_csv(path,index=False)

def make_pair(tr,va,cfg,first):
    selector=FeatureSelector().fit(tr) # never fit on validation experiment
    a=windows(tr,selector,cfg['history'],cfg['horizon'],min_target_row=first,reconstructed_weight=cfg['weight'])
    b=windows(va,selector,cfg['history'],cfg['horizon'],min_target_row=first)
    scales,noise=scales_from_training(a,tr)
    return a,b,scales,noise,selector

def observed_score(pred,d):
    per=[]
    for j in range(1,6):
        m=d['observed'][:,j].astype(bool)
        if not m.any():continue
        diff=pred[m,j]-d['y'][m,j]
        per.append(dict(location=f'L{j+1}',n=int(m.sum()),rmse=float(np.sqrt(np.mean(diff**2))),mae=float(np.mean(np.abs(diff))),bias=float(np.mean(diff))))
    return per

def macro(rows,key='mae'):
    vals=[r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float('inf')

def experiment_macro(rows,metric='mae'):
    # Equal weight per held-out experiment; within experiment, equal weight per location.
    experiments=sorted(set(r['held'] for r in rows))
    return float(np.mean([macro([r for r in rows if r['held']==held],metric) for held in experiments])) if experiments else float('inf')

def score_methods(pred,d):
    out={'transformer':observed_score(pred,d)}
    for method,key,availkey in [('persistence','baseline','baseline_available'),('two_dwell_mean','smoothed','smoothed_available')]:
        mask=d['observed'].astype(bool)&d[availkey].astype(bool)
        mask[:,0]=False
        paired=mask.copy()
        out[method]=observed_score(d[key],{**d,'observed':paired})
        out['transformer_paired_'+method]=observed_score(pred,{**d,'observed':paired})
    return out

def score_for_selection(pred,d,metric,active,scales):
    from co2_submission.diagnostics import metric_rows
    if metric in ('mae','rmse'):
        scores=[r[metric] for r in observed_score(pred,d)]
        return float(np.mean(scores)) if scores else float('inf')
    vals,_=metric_rows(pred,d,scales,'inner_validation',0,active)
    return float(vals['observed_'+metric])

def run_fit(tr,va,cfg,first,device,epochs,checkpoint_epochs=None,save_dir=None,fold='',patience=None,min_delta=0.0,selection_metric="mae"):
    selector=FeatureSelector().fit(tr)
    a=windows(tr,selector,cfg['history'],cfg['horizon'],min_target_row=first,reconstructed_weight=cfg['weight'])
    b=windows(va,selector,cfg['history'],cfg['horizon'],min_target_row=first) if va is not None else None
    scales,noise=scales_from_training(a,tr)
    model,curves,locations,checkpoints=train(a,scales,epochs=epochs,seed=cfg['seed'],device=device,
        kind=cfg['loss'],mode=cfg['mode'],reconstructed_weight=cfg['weight'],validation=b,
        checkpoint_epochs=checkpoint_epochs,lr=cfg.get('lr',1e-3),l1_proxy_weight=cfg.get('l1_proxy_weight',.05),
        patience=patience,min_delta=min_delta,selection_metric=selection_metric)
    if save_dir is not None:
        save_dir.mkdir(parents=True,exist_ok=True)
        save(curves,save_dir/'epoch_metrics.csv');save(locations,save_dir/'location_objective_metrics.csv')
        (save_dir/'statistics.json').write_text(json.dumps({'noise':noise,'features':selector.summary(),'scales':{k:v.tolist() for k,v in scales.items()},'config':cfg},indent=2))
    return model,curves,locations,checkpoints,b,noise,selector,scales

def checkpoint_score(state,b,cfg,device):
    model=CO2Forecaster(mode=cfg['mode']).to(device)
    model.load_state_dict(state)
    pred=predict(model,b,device)
    return observed_score(pred,b),pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',default='data/raw');ap.add_argument('--out',default='outputs/direct')
    ap.add_argument('--mode',choices=['cv','final-test','audit'],default='cv')
    ap.add_argument('--histories',nargs='+',type=int,default=[18]);ap.add_argument('--horizons',nargs='+',type=int,default=[3])
    ap.add_argument('--losses',nargs='+',choices=['raw','variance','noise'],default=['raw','variance','noise'])
    ap.add_argument('--input-modes',nargs='+',choices=['combined','process','co2'],default=['combined'])
    ap.add_argument('--weights',nargs='+',type=float,default=[.05]);ap.add_argument('--l1-proxy-weight',type=float,default=.05)
    ap.add_argument('--max-epochs',type=int,default=150,help='Maximum epochs per inner fit with patience stopping')
    ap.add_argument('--patience',type=int,default=20,help='Consecutive non-improving epochs before stopping')
    ap.add_argument('--min-delta',type=float,default=0.0001,help='Minimum improvement in observed validation metric')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--config')
    ap.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    ap.add_argument('--torch-threads',type=int,default=1,help='CPU intra-op threads; default 1 prevents severe oversubscription on small batches')
    ap.add_argument('--overwrite',action='store_true',help='Explicitly allow overwriting existing outputs')
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--selection-metric',choices=['mae','rmse'],default='mae')
    ap.add_argument('--allow-retrospective-labels',action='store_true',help='Explicitly acknowledge offline, retrospectively assigned analyzer location labels')
    ap.add_argument('--folds',nargs='*',default=None,help='Optional development folds for smoke testing; not for final model selection')
    args=ap.parse_args();out=Path(args.out)
    if args.torch_threads<1:raise ValueError('torch-threads must be >=1')
    torch.set_num_threads(args.torch_threads)
    # Refuse any preexisting artifact directory unless the user explicitly opts in.
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out}. Use a new --out or --overwrite.')
    if args.overwrite and args.mode=='final-test' and out.exists() and any(out.iterdir()):
        raise FileExistsError('Final-test overwrite is intentionally prohibited. Use a new empty --out.')
    out.mkdir(parents=True,exist_ok=True)
    if any(h<1 for h in args.histories) or any(h<1 for h in args.horizons):raise ValueError('History and horizon must be >=1')
    if not 0<=args.l1_proxy_weight<=1:raise ValueError('L1 proxy weight must be in [0,1]')
    if args.max_epochs<1 or args.patience<1 or args.min_delta<0:raise ValueError('Invalid early stopping parameters')
    if args.mode=='cv' and not args.allow_retrospective_labels and any(m!='process' for m in args.input_modes):
        raise ValueError('CO2-history inputs require --allow-retrospective-labels: source labels are not verified causally. Use --input-modes process for strictly causal process-only study.')
    runs=load_runs(args.raw_dir);test='140207_1.xlsx'
    if test not in runs:raise ValueError('Final-test experiment absent')
    dev={k:v for k,v in runs.items() if k!=test}
    if len(dev)!=7:raise ValueError('Expected seven development experiments')
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'seed':args.seed},indent=2))
    device=args.device
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS not available')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    if args.mode=='audit':
        cfg=dict(history=18,horizon=3,loss='noise',mode='combined',weight=.05,seed=args.seed,lr=args.lr)
        a,_,scales,noise,sel=make_pair({k:v for k,v in dev.items() if k!='140120_1.xlsx'},{'140120_1.xlsx':dev['140120_1.xlsx']},cfg,40)
        (out/'noise_audit.json').write_text(json.dumps({'noise':noise,'scales':{k:v.tolist() for k,v in scales.items()}},indent=2));return
    if args.mode=='final-test':
        if not args.config:raise ValueError('--config required')
        if not args.overwrite and any((out / x).exists() for x in ('final_predictions.csv','final_metrics.csv','final_model.pt')):
            raise FileExistsError('Final test artifacts exist. Choose a new --out or supply --overwrite.')
        cfg=json.loads(Path(args.config).read_text());
        if cfg['mode']!='process' and not args.allow_retrospective_labels:raise ValueError('Final test requires explicit --allow-retrospective-labels for CO2 inputs')
        first=int(cfg['first_target_row']);epoch=int(cfg['epoch'])
        model,curves,locations,_,_,noise,selector,scales=run_fit(dev,None,cfg,first,device,epoch,save_dir=out/'final_training',fold=test)
        b=windows({test:runs[test]},selector,cfg['history'],cfg['horizon'],min_target_row=first)
        pred=predict(model,b,device)
        save(prediction_rows(pred,b,'final_test',cfg,test,epoch),out/'final_predictions.csv')
        save([dict(method=method,**r) for method,rows in score_methods(pred,b).items() for r in rows],out/'final_metrics.csv')
        torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'config':cfg,'features':selector.summary(),'noise':noise,'scales':{k:v.tolist() for k,v in scales.items()}},out/'final_model.pt')
        print('Final test written:',out);return
    if any(w<0 or w>1 for w in args.weights):raise ValueError('Reconstruction weights must be in [0,1]')
    # Each configuration uses its own minimum feasible target row; no arbitrary 40-row cutoff.
    first=None
    folds=sorted(dev) if args.folds is None else args.folds
    if any(f not in dev for f in folds):raise ValueError('Unknown development fold')
    if args.folds is not None:print('WARNING: partial folds requested; output is diagnostic only, not final model selection',flush=True)
    allrows=[];config_results=[];allpred=[];epoch_rows=[];loc_rows=[];coverage_rows=[];missing_rows=[];noise_rows=[];scale_rows=[];early_rows=[]
    for h,hor,loss,mode,w in product(args.histories,args.horizons,args.losses,args.input_modes,args.weights):
        cfg=dict(history=h,horizon=hor,loss=loss,mode=mode,weight=w,l1_proxy_weight=args.l1_proxy_weight,seed=args.seed,lr=args.lr)
        first=h-1+hor
        cfgid=f'h{h}_f{hor}_{loss}_{mode}_w{w:g}'
        for held in folds:
            remaining={k:v for k,v in dev.items() if k!=held}
            inner_best=[]
            for innerheld in sorted(remaining):
                itr={k:v for k,v in remaining.items() if k!=innerheld}
                try:
                    model,curve,loc,checkpoints,b,_,_,scales=run_fit(
                        itr,{innerheld:remaining[innerheld]},cfg,first,device,args.max_epochs,
                        patience=args.patience,min_delta=args.min_delta,selection_metric=args.selection_metric)
                except ValueError as exc:
                    raise RuntimeError(f'Inner fold {innerheld} failed; abort rather than silently bias CV') from exc
                for r in curve:epoch_rows.append(dict(config_id=cfgid,outer_fold=held,inner_fold=innerheld,**r))
                for r in loc:loc_rows.append(dict(config_id=cfgid,outer_fold=held,inner_fold=innerheld,**r))
                es=checkpoints['early_stopping']
                inner_best.append(int(es['best_epoch']))
                early_rows.append(dict(config_id=cfgid,outer_fold=held,inner_fold=innerheld,
                                       best_epoch=es['best_epoch'],best_score=es['best_score'],
                                       stopped_epoch=es['stopped_epoch'],stop_reason=es['stop_reason']))
            # Deterministic common duration from inner best epochs only. No outer labels.
            # Median reduces influence of one unusually early/late validation optimum.
            chosen=int(np.floor(np.median(inner_best)+0.5))
            fold_dir=out/'outer_folds'/cfgid/held.replace('.xlsx','')
            model,curve,loc,_,_,noise,selector,scales=run_fit(remaining,None,cfg,first,device,chosen,save_dir=fold_dir)
            b=windows({held:dev[held]},selector,cfg['history'],cfg['horizon'],min_target_row=first)
            pred=predict(model,b,device)
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'config':cfg,'epoch':chosen,'features':selector.summary(),
                        'noise':noise,'scales':{k:v.tolist() for k,v in scales.items()}},fold_dir/'model.pt')
            for name,df in remaining.items():
                for r in selector.missingness(df):missing_rows.append(dict(config_id=cfgid,held=held,experiment=name,**r))
            for r in interpolation_diagnostics(dev[held],selector.columns):coverage_rows.append(dict(config_id=cfgid,held=held,**r))
            # Report actual coverage and how many feasible target rows were excluded.
            for experiment,df in {held:dev[held]}.items():
                from co2_submission.reconstruction import targets
                _,valid_mask,obs_mask,_=targets(df,True)
                feasible=max(0,len(df)-first)
                included=int(valid_mask[first:,1:].any(axis=1).sum())
                observed_rows=int(obs_mask[first:,1:].any(axis=1).sum())
                coverage_rows.append(dict(config_id=cfgid,held=held,experiment=experiment,
                    location='ALL_L2_L6',diagnostic_type='coverage',feasible_target_rows=feasible,
                    included_target_rows=included,excluded_no_valid_target=feasible-included,
                    observed_target_rows=observed_rows,reconstruction_only_target_rows=included-observed_rows))
            for r in noise['noise']:noise_rows.append(dict(config_id=cfgid,held=held,**r))
            for j in range(1,6):
                scale_rows.append(dict(config_id=cfgid,held=held,location=f'L{j+1}',raw_scale=float(scales['raw'][j]),variance_scale=float(scales['variance'][j]),noise_scale=float(scales['noise'][j]),noise_over_variance=float(scales['noise'][j]/scales['variance'][j]),effectively_equal=bool(np.isclose(scales['noise'][j],scales['variance'][j]))))
            allpred.extend(prediction_rows(pred,b,'outer_test',cfg,held,chosen))
            state_rows=[dict(config_id=cfgid,held=held,experiment=str(b['run'][ii]),origin_row=int(b['origin'][ii]),
                             target_row=int(b['target'][ii]),**{f'process_scaled_{col}':float(b['process_origin'][ii,j]) for j,col in enumerate(selector.columns)})
                        for ii in range(len(b['x']))]
            save(state_rows,fold_dir/'operating_state.csv')
            for method,rows in score_methods(pred,b).items():
                for r in rows:allrows.append(dict(config_id=cfgid,held=held,epoch=chosen,method=method,**r))
            print('outer',held,cfgid,'epoch',chosen,'macro MAE',macro(observed_score(pred,b)),flush=True)
            save(allrows,out/'cv_folds.csv');save(allpred,out/'outer_predictions.csv')
            save(epoch_rows,out/'inner_epoch_metrics.csv');save(loc_rows,out/'inner_location_objective_metrics.csv');save(early_rows,out/'early_stopping_summary.csv')
            save(coverage_rows,out/'interpolation_gaps.csv');save(missing_rows,out/'feature_missingness.csv');save(noise_rows,out/'noise_estimates.csv');save(scale_rows,out/'objective_scale_comparison.csv')
        subset=[r for r in allrows if r['config_id']==cfgid and r['method']=='transformer']
        config_results.append(dict(config_id=cfgid,**cfg,selection_metric=args.selection_metric,macro_observed_mae=experiment_macro(subset),macro_observed_rmse=experiment_macro(subset,'rmse')))
        save(config_results,out/'cv_summary.csv')
    if args.folds is not None:
        print('Partial-fold diagnostics saved; selected_config.json intentionally NOT produced');return
    # Outer CV compares configurations; selected outer score is NOT an unbiased performance estimate.
    # Final frozen test remains the independent evaluation.
    best=min(config_results,key=lambda r:r['macro_observed_mae'] if args.selection_metric=='mae' else r['macro_observed_rmse'])
    cfg={k:best[k] for k in ('history','horizon','loss','mode','weight','l1_proxy_weight','seed','lr')}
    first=cfg['history']-1+cfg['horizon']  # Selected configuration, not last grid candidate.
    final_inner_best=[]
    for held in sorted(dev):
        tr={k:v for k,v in dev.items() if k!=held}
        model,curve,loc,checkpoints,b,_,_,scales=run_fit(
            tr,{held:dev[held]},cfg,first,device,args.max_epochs,
            patience=args.patience,min_delta=args.min_delta,selection_metric=args.selection_metric)
        es=checkpoints['early_stopping']
        final_inner_best.append(int(es['best_epoch']))
        early_rows.append(dict(config_id='final_epoch_selection',outer_fold='ALL_DEVELOPMENT',inner_fold=held,
                               best_epoch=es['best_epoch'],best_score=es['best_score'],
                               stopped_epoch=es['stopped_epoch'],stop_reason=es['stop_reason']))
        for r in curve:epoch_rows.append(dict(config_id='final_epoch_selection',outer_fold='ALL_DEVELOPMENT',inner_fold=held,**r))
    save(early_rows,out/'early_stopping_summary.csv')
    save(epoch_rows,out/'inner_epoch_metrics.csv')
    cfg['epoch']=int(np.floor(np.median(final_inner_best)+0.5))
    cfg['early_stopping']=dict(max_epochs=args.max_epochs,patience=args.patience,min_delta=args.min_delta,
                               epoch_rule='rounded_median_of_inner_fold_best_epochs')
    cfg['selection_metric']=args.selection_metric
    cfg['retrospective_location_labels_assumed']=bool(args.allow_retrospective_labels)
    cfg['first_target_row']=cfg['history']-1+cfg['horizon']
    cfg['outer_cv_selection_bias_warning']='Configuration chosen using outer development CV; do not report selected outer CV score as unbiased.'
    (out/'selected_config.json').write_text(json.dumps(cfg,indent=2))
    print('Selected on development only:',cfg)
if __name__=='__main__':main()
