"""Read labeled OA sheet; second worksheet row contains units, not observations."""
from pathlib import Path
import pandas as pd

def load_runs(directory,exclude=None):
    runs={}
    for path in sorted(Path(directory).glob('*.xlsx')):
        if path.name.startswith('~$') or path.name in (exclude or set()):continue
        df=pd.read_excel(path,sheet_name=0,skiprows=[1]).rename(columns=lambda x:str(x).strip())
        df=df.rename(columns={df.columns[0]:'timestamp'})
        df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce')
        if df.timestamp.isna().any() or not df.timestamp.is_monotonic_increasing:raise ValueError(f'Invalid time in {path}')
        for c in df.columns.drop('timestamp'):df[c]=pd.to_numeric(df[c],errors='coerce')
        if not df.label.isin(range(1,7)).all():raise ValueError(f'Invalid label in {path}')
        if df.AT400.isna().any():raise ValueError(f'Missing analyzer values in {path}')
        key=path.stem.split('(')[0]+'.xlsx'
        if key in runs:raise ValueError(f'Duplicate experiment ID {key}: {path}')
        runs[key]=df.reset_index(drop=True)
    if not runs:raise ValueError('No experiments')
    if len(runs)!=len(set(runs)):raise ValueError('Duplicate experiment IDs')
    return runs
