"""Training-only empirical analyzer noise from within-dwell consecutive differences.
For independent errors, MAD(diff)/sqrt(2) estimates per-reading noise.
Stable dwells chosen using TRAINING runs only; report counts and fallback.
"""
import numpy as np

def noise_estimates(runs,min_differences=5):
    byloc=[[] for _ in range(6)]; counts=[0]*6
    for df in runs.values():
        labels=df.label.to_numpy(int); values=df.AT400.to_numpy(float)
        start=0
        for end in range(1,len(df)+1):
            if end<len(df) and labels[end]==labels[start]:continue
            j=labels[start]-1
            if j>=1:
                v=values[start:end];v=v[np.isfinite(v)]
                if len(v)>=2:
                    # Consecutive differences within dwell, excluding first transition reading.
                    d=np.diff(v[1:]) if len(v)>=3 else np.array([])
                    if len(d):byloc[j].extend(d.tolist());counts[j]+=len(d)
            start=end
    scales=[];details=[]
    for j in range(6):
        d=np.asarray(byloc[j]);mad=1.4826*np.median(np.abs(d-np.median(d)))/np.sqrt(2) if len(d) else np.nan
        # A noise estimate is not identifiable from too few differences or a degenerate MAD.
        valid= len(d)>=min_differences and np.isfinite(mad) and mad>0
        scales.append(float(mad) if valid else None)
        details.append({'location':j+1,'differences':len(d),'noise_sigma':float(mad) if np.isfinite(mad) else None,'identifiable':bool(valid)})
    return scales,details

def scales_from_training(data,runs):
    noise,detail=noise_estimates(runs)
    robust=[]
    for j in range(6):
        z=data['y'][data['observed'][:,j],j]
        r=float(np.subtract(*np.percentile(z,[75,25]))/1.349) if len(z)>3 else 0.
        robust.append(max(r,1e-4))
    # Floor based on robust scale of informative L3-L6, not tiny L2 variance.
    floor=max(.05*float(np.median(robust[2:6])),1e-3)
    # empirical noise floor where identifiable, otherwise conservative shared floor
    empirical=np.asarray([max(x or 0,floor) for x in noise],np.float32)
    robust_floor=np.asarray([max(r,floor) for r in robust],np.float32)
    # L1 proxy is derived from L2, so use the L2 residual scale for its weak loss.
    robust_floor[0]=robust_floor[1]
    empirical[0]=empirical[1]
    return {'raw':np.ones(6,np.float32),'variance':robust_floor,'noise':np.maximum(robust_floor,3*empirical)}, {'noise':detail,'robust_scales':robust,'floor':floor,'effective_noise_scales':np.maximum(robust_floor,3*empirical).tolist()}
