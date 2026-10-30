"""Causal CO2 event streams. A dwell is available only AFTER its last row.
Labels in provided spreadsheets are retrospective; use only with explicit offline assumption.
"""
import numpy as np

def context(df):
    n=len(df); labels=df.label.to_numpy(int); values=df.AT400.to_numpy(float)
    events=np.zeros((n,6),np.float32); state=np.zeros((n,6),np.float32)
    availability=np.zeros((n,6),np.float32); age=np.zeros((n,6),np.float32)
    last=np.full(6,np.nan); last_at=np.full(6,-1)
    prev=np.full(6,np.nan); previous_count=np.zeros(6,int)
    smooth=np.zeros((n,6),np.float32);smooth_avail=np.zeros((n,6),bool)
    start=0
    for t in range(n):
        if t>0 and labels[t]!=labels[t-1]:
            j=labels[t-1]-1
            if j>=1:  # L1 analyzer is unreliable: no L1 physical event
                v=values[start:t]; v=v[np.isfinite(v)]
                if len(v):
                    prev[j]=last[j];last[j]=float(v.mean());last_at[j]=t-1;events[t,j]=1
                    previous_count[j]+=1
            start=t
        # one-hot is only for completed, labeled dwell; never current incomplete dwell
        for j in range(1,6):
            if last_at[j]>=0:
                state[t,j]=last[j];availability[t,j]=1;age[t,j]=t-last_at[j]
                if previous_count[j]>=2:
                    smooth[t,j]=(last[j]+prev[j])/2;smooth_avail[t,j]=True
    # 24 distinct historical features: values, new-dwell events, availability, age.
    age=np.where(availability>0,np.log1p(age),0).astype(np.float32)
    return np.concatenate([state,events,availability,age],axis=1),availability,smooth,smooth_avail,age
