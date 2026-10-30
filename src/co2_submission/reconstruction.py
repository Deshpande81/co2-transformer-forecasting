"""Retrospective targets, never historical model inputs.
L1 uses preceding completed L2 dwell as proxy; no genuine L1 target.
"""
import numpy as np
import pandas as pd

def targets(df,use_reconstruction=True):
    n=len(df);loc=df.label.to_numpy(int);co2=df.AT400.to_numpy(float)
    y=np.full((n,6),np.nan);observed=np.zeros((n,6),bool);proxy=np.zeros((n,6),bool)
    for j in range(1,6):
        m=(loc==j+1)&np.isfinite(co2);y[m,j]=co2[m];observed[m,j]=True
    if use_reconstruction:
        for j in range(1,6):
            y[:,j]=pd.Series(y[:,j]).interpolate(limit_area='inside').to_numpy()
    # Paper-inspired L1 surrogate: L1 is not an independent trustworthy reading.
    # Use the contemporaneous L2 reference profile as the L1 proxy. This is
    # retrospective target construction ONLY; it is never passed as an input.
    # In particular, do not use an unreliable physical L1 analyzer reading.
    # Paper §2.1/§2.3: contaminated L1 analyzer readings are replaced by
    # the average of L2 readings, not by the pointwise interpolated L2 curve.
    # A sampling cycle visits L1 then L2. For each completed L1 dwell,
    # assign the mean of the immediately following L2 dwell. The resulting
    # L1 cycle anchors are interpolated *only as retrospective targets*.
    # If no following L2 dwell exists, the L1 dwell remains unlabelled.
    starts=np.r_[0, np.flatnonzero(loc[1:]!=loc[:-1])+1]
    ends=np.r_[starts[1:], n]
    l1_anchors=np.full(n,np.nan,dtype=float)
    for k,(start,end) in enumerate(zip(starts,ends)):
        if loc[start]!=1 or k+1>=len(starts):
            continue
        next_start,next_end=starts[k+1],ends[k+1]
        if loc[next_start]!=2:
            continue
        readings=co2[next_start:next_end]
        readings=readings[np.isfinite(readings)]
        if len(readings):
            l1_anchors[start:end]=float(readings.mean())
    if use_reconstruction:
        y[:,0]=pd.Series(l1_anchors).interpolate(limit_area='inside').to_numpy()
    else:
        y[:,0]=l1_anchors
    proxy[:,0]=np.isfinite(y[:,0])
    valid=np.isfinite(y)
    return np.nan_to_num(y,nan=0).astype('float32'),valid,observed,proxy


def interpolation_diagnostics(df,feature_columns=None):
    """Describe interpolation intervals without changing reference reconstruction."""
    _, valid, observed, proxy = targets(df,True)
    rows=[]
    feature_columns=feature_columns or []
    # Process-change statistic is diagnostic only: never used as a target/input.
    proc=df[feature_columns].replace([np.inf,-np.inf],np.nan).ffill() if feature_columns else None
    for j in range(1,6):
        inds=np.flatnonzero(observed[:,j])
        for left,right in zip(inds[:-1],inds[1:]):
            if right-left<=1:continue
            process_change=np.nan
            if proc is not None:
                left_vals=proc.iloc[left].to_numpy(float)
                right_vals=proc.iloc[right].to_numpy(float)
                diff=np.abs(right_vals-left_vals)
                process_change=float(np.nanmedian(diff)) if np.isfinite(diff).any() else np.nan
            rows.append(dict(location=f'L{j+1}',left_row=int(left),right_row=int(right),
                             median_absolute_process_endpoint_change=process_change,
                             gap_rows=int(right-left-1),
                             max_interpolation_distance_rows=int((right-left)//2),
                             experiment_rows=len(df)))
    return rows
