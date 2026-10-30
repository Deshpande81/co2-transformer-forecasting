import numpy as np
from .context import context
from .reconstruction import targets

def windows(runs,selector,history,horizon,min_target_row=None,reconstruct=True,reconstructed_weight=.05):
    keys=('x','y','valid','observed','proxy','run','origin','target','baseline','baseline_available','smoothed','smoothed_available','measurement_age','process_origin','target_observed_any','target_reconstructed_any')
    out={k:[] for k in keys}
    first=max(history-1+horizon,min_target_row or 0)
    for name,df in runs.items():
        proc=selector.transform(df);co2,available,smooth,smooth_avail,ages=context(df)
        y,valid,obs,proxy=targets(df,reconstruct)
        for target in range(first,len(df)):
            origin=target-horizon
            # Every timestamp with at least one valid L2-L6 target participates.
            # Evaluation population must not change with reconstruction weight.
            # Training may retain all such windows, even when reconstruction is disabled.
            if not valid[target,1:].any():continue
            hist=np.concatenate([proc[origin-history+1:origin+1],co2[origin-history+1:origin+1]],axis=1)
            rec=dict(x=hist,y=y[target],valid=valid[target],observed=obs[target],proxy=proxy[target],run=name,origin=origin,target=target,baseline=co2[origin,0:6],baseline_available=available[origin],smoothed=smooth[origin],smoothed_available=smooth_avail[origin],measurement_age=ages[origin],process_origin=proc[origin],target_observed_any=bool(obs[target,1:].any()),target_reconstructed_any=bool((valid[target,1:] & ~obs[target,1:]).any()))
            for k in keys:out[k].append(rec[k])
    if not out['x']:raise ValueError('No valid target windows')
    return {k:np.asarray(v) for k,v in out.items()}
