import numpy as np
import pytest
from co2_submission.training import train

def sample():
    rng=np.random.default_rng(19);n=24
    y=np.abs(rng.normal(size=(n,6))).astype('float32')
    data={'x':rng.normal(size=(n,4,54)).astype('float32'), 'y':y,
          'valid':np.ones((n,6),bool),'observed':np.ones((n,6),bool)}
    scales={k:np.ones(6,dtype='float32') for k in ('raw','variance','noise')}
    return data,scales

@pytest.mark.parametrize('kind',['raw','variance','noise'])
def test_early_stopping_all_objectives(kind):
    data,scales=sample()
    model,curve,loc,checkpoints=train(data,scales,epochs=4,kind=kind,mode='combined',
         validation=data,patience=2,min_delta=1000,selection_metric='mae')
    info=checkpoints['early_stopping']
    assert info['best_epoch']==1
    assert info['stopped_epoch']==3
    assert info['stop_reason']=='patience'
    assert [r['epoch'] for r in curve if r['split']=='inner_validation']==[1,2,3]
    assert all('selection_score' in r for r in curve if r['split']=='inner_validation')
    assert all(np.array_equal(model.state_dict()[k].cpu().numpy(),v.numpy()) for k,v in info['state_dict'].items())

def test_max_epoch_reports_no_patience_trigger():
    data,scales=sample()
    _,curve,_,ck=train(data,scales,epochs=2,kind='raw',mode='combined',validation=data,
                      patience=5,min_delta=0,selection_metric='rmse')
    assert ck['early_stopping']['stop_reason']=='max_epochs'
    assert len([r for r in curve if r['split']=='inner_validation'])==2

def test_early_stopping_requires_validation():
    data,scales=sample()
    with pytest.raises(ValueError,match='requires validation'):
        train(data,scales,epochs=1,patience=2)
