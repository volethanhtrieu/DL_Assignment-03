import sys
import json
import pathlib
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from summarize_dev import summarize


def test_no_synthetic_metrics_can_enter_report(tmp_path):
    runs=tmp_path/'runs'
    for name in ('image','text','concat'):
        for seed in (42,43,44):
            p=runs/name/f'seed_{seed}'
            p.mkdir(parents=True)
            (p/'run.json').write_text(json.dumps({'model':name,'seed':seed,'smoke_only':True,
                'dev':{'macro_f1':.123,'accuracy':.5},'train_and_dev_seconds':1,
                'best_epoch':1,'trainable_parameters':100}))
            (p/'best.pt').touch();(p/'dev_predictions.csv').touch()
    with pytest.raises(ValueError,match='Synthetic'):
        summarize(runs,tmp_path/'out','m2')
