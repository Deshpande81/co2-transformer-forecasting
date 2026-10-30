import numpy as np
from co2_submission.diagnostics import metric_rows,prediction_rows

def test_all_objective_metrics_and_location_shares():
    n=8
    data={'y':np.ones((n,6),dtype='float32'),'valid':np.ones((n,6),bool),'observed':np.ones((n,6),bool)}
    scales={'raw':np.ones(6),'variance':np.full(6,2.),'noise':np.full(6,3.)}
    rows,loc=metric_rows(np.zeros((n,6),dtype='float32'),data,scales,'inner_validation',1,'raw')
    assert all('observed_'+k+'_huber' in rows for k in scales)
    assert len(loc)==15
    for kind in scales:
        assert abs(sum(r['loss_share_pct'] for r in loc if r['evaluated_objective']==kind)-100)<1e-4

def test_prediction_export_has_targets():
    data={'y':np.ones((1,6)),'observed':np.ones((1,6),bool),'valid':np.ones((1,6),bool),'baseline':np.ones((1,6)),
          'baseline_available':np.ones((1,6),bool),'smoothed':np.ones((1,6)),'smoothed_available':np.ones((1,6),bool),'measurement_age':np.zeros((1,6)),'run':np.array(['run']),'origin':np.array([3]),'target':np.array([5])}
    cfg={'history':2,'horizon':2,'mode':'combined','loss':'noise'}
    rows=prediction_rows(np.ones((1,6)),data,'outer_test',cfg,'fold',1)
    assert len(rows)==6 and 'y_true' in rows[0] and 'observed' in rows[0]
