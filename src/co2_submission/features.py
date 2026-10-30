"""Fixed, auditable 30-sensor process feature set; fit statistics on training runs only."""
import numpy as np
import pandas as pd
FEATURES = ('AT100','AT300','FT100','FT102','FT103','FT104','FT105','FT106','FT108','FT109','FT116','FT201','FT301','FT302','FT303','FT304','FT305','FT402','LT101','LT104','PDT105','PT110','PT402','PT403','TT100','TT102','TT200','TT210','TT211','TT213')
assert len(FEATURES)==30 and len(set(FEATURES))==30
class FeatureSelector:
    def __init__(self,n=30):
        if n!=30: raise ValueError('Exactly 30 fixed features required')
    def _clean(self,df):
        return df[self.columns].replace([np.inf,-np.inf],np.nan)
    def fit(self,runs):
        if not runs: raise ValueError('Empty training runs')
        self.columns=list(FEATURES)
        for name,df in runs.items():
            missing=[c for c in FEATURES if c not in df]
            if missing: raise ValueError(f'{name} missing sensors {missing}')
        raw=pd.concat([self._clean(df) for df in runs.values()],ignore_index=True)
        med=raw.median().fillna(0)
        self.median=med.to_numpy(float)
        # Match the transformation at inference: causal ffill independently per run,
        # followed by TRAINING-only medians for leading missing observations.
        filled=pd.concat([self._clean(df).ffill().fillna(med) for df in runs.values()],ignore_index=True)
        self.mean=filled.mean().to_numpy(float)
        self.std=np.maximum(filled.std(ddof=0).to_numpy(float),1e-6)
        return self
    def transform(self,df):
        if not hasattr(self,'median'):raise RuntimeError('FeatureSelector must be fitted')
        x=self._clean(df).ffill().fillna(pd.Series(self.median,index=self.columns))
        return ((x.to_numpy(float)-self.mean)/self.std).astype('float32')
    def missingness(self,df):
        x=df[self.columns].replace([np.inf,-np.inf],np.nan)
        rows=[]
        for c in self.columns:
            missing=x[c].isna().to_numpy()
            longest=0; current=0
            for m in missing:
                current=current+1 if m else 0;longest=max(longest,current)
            rows.append({'feature':c,'missing_count':int(missing.sum()),
                         'missing_fraction':float(missing.mean()),'max_consecutive_missing_rows':longest})
        return rows
    def summary(self):return {'selected_features':self.columns,'train_median':self.median.tolist(),'train_mean':self.mean.tolist(),'train_std':self.std.tolist()}
