import sys
import pathlib
import numpy as np
import pytest
import torch

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from experiment import CLASSES, Head, MAX_TOKENS, MODELS, normalise_text, score, load_features, run_smoke


def test_all_models_predict_five_logits():
    a=torch.randn(3,512);b=torch.randn(3,768)
    for variant in MODELS:
        m=Head(variant).eval()
        y,g=m(a,b,return_gate=True)
        assert y.shape==(3,5)
        if variant in ('gated','fixed_gate'):
            assert g.shape==(3,1) and bool(((g>=0)&(g<=1)).all())
        else: assert g is None


def test_gate_ablation_has_fewer_parameters():
    full=sum(x.numel() for x in Head('gated').parameters())
    fixed=sum(x.numel() for x in Head('fixed_gate').parameters())
    assert full-fixed==257


def test_source_normalisation_matches_m1():
    assert normalise_text('RT @A: FLOOD &amp; Help https://t.co/123 @ABC')=='flood help'
    assert normalise_text('Café A')=='café a'


def test_metrics_include_absent_classes():
    out=score(np.array([4,4,4]),np.array([4,4,4]))
    assert out['accuracy']==1.0 and out['macro_f1']==pytest.approx(0.2)
    assert out['per_class'][CLASSES[0]]['support']==0


def test_no_pickle_and_shape_integrity(tmp_path):
    p=tmp_path/'train.npz'
    np.savez_compressed(p,row_ids=np.array(['train:x']),group_ids=np.array(['g1']),
                        y=np.array([0]),image=np.ones((1,512),dtype='float32'),
                        text=np.ones((1,768),dtype='float32'))
    d=load_features(p)
    assert d['image'].shape==(1,512)
    with pytest.raises(AssertionError):load_features(p,expected_split='train')


def test_synthetic_end_to_end_smoke(tmp_path):
    outcome=run_smoke(tmp_path/'smoke')
    assert len(outcome)==5
    for variant in MODELS:
        assert (tmp_path/'smoke'/'runs'/variant/'seed_42'/'best.pt').is_file()
    assert (tmp_path/'smoke'/'SMOKE_ONLY.json').is_file()
