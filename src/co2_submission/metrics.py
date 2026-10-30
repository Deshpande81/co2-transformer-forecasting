import numpy as np

def evaluate(pred,y,mask):
    out={}
    for j in range(6):
        m=mask[:,j];d=pred[m,j]-y[m,j]
        out[f'L{j+1}_n']=int(m.sum());out[f'L{j+1}_rmse']=float(np.sqrt(np.mean(d*d))) if len(d) else None
    for label,indices in [('L2_L5',range(1,5)),('L2_L6',range(1,6)),('all_six',range(6))]:
        m=mask[:,list(indices)];d=(pred[:,list(indices)]-y[:,list(indices)])[m]
        out[label+'_n']=int(m.sum());out[label+'_rmse']=float(np.sqrt(np.mean(d*d))) if len(d) else None
        vals=[out[f'L{j+1}_rmse'] for j in indices if out[f'L{j+1}_rmse'] is not None]
        out[label+'_macro_rmse']=float(np.mean(vals)) if vals else None
    return out
