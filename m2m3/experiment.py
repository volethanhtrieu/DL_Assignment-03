#!/usr/bin/env python3
"""CO3133 A3 frozen-encoder experiment suite. Requires real locked CrisisMMD bytes.

No fabricated results. The 'smoke' command only tests model plumbing on SYNTHETIC data.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import io
import json
import os
import pathlib
import platform
import random
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = (
    'affected_individuals', 'infrastructure_and_utility_damage',
    'rescue_volunteering_or_donation_effort', 'other_relevant_information',
    'not_humanitarian',
)
SPLITS = ('train', 'dev', 'test')
COUNTS = {'train': 6001, 'dev': 986, 'test': 951}
MODELS = ('image', 'text', 'concat', 'gated', 'fixed_gate')
SEEDS = (42, 43, 44)
IMAGE_DIM, TEXT_DIM, HIDDEN, NUM_CLASSES = 512, 768, 128, 5
TEXT_MODEL = 'distilbert/distilbert-base-uncased'
TEXT_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'  # HF model repo commit, 2024-05-06
MAX_TOKENS = 128


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalise_text(s: str) -> str:
    """Exactly the source-audit normalisation in A3-M1/scripts/audit.py."""
    s = unicodedata.normalize('NFKC', html.unescape(s)).casefold()
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'^\s*rt\s+@\w+\s*:\s*', '', s)
    s = re.sub(r'@\w+', ' ', s)
    return ' '.join(re.findall(r'\w+', s))


def read_csv(p: pathlib.Path) -> list[dict]:
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def dump_json(p: pathlib.Path, obj: dict | list):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def validate_locked_repository(repo: pathlib.Path) -> tuple[dict, dict[str, list[dict]]]:
    lock_path = repo / 'A3-M1/DATASET_LOCK.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert lock['dataset_version'] == 'crisismmd5-mirror-remediation-20261104-v1'
    assert lock['primary_protocol'] == 'filtered_original'
    assert lock['split_counts'] == COUNTS
    for relative_path, fingerprint in lock['files'].items():
        p = repo / relative_path
        if not p.is_file() or sha256(p) != fingerprint:
            raise ValueError(f'Locked input missing/changed: {relative_path}')
    manifest_dir = repo / 'A3-M1/remediation/manifests/filtered_original'
    manifests = {s: read_csv(manifest_dir / (s + '.csv')) for s in SPLITS}
    seen_rows = set()
    unique_by_key = {key: {} for key in ('tweet_id','image_sha256','pixel_sha256','raw_text_sha256','normalized_text_sha256','group_id')}
    for split, rows in manifests.items():
        assert len(rows) == COUNTS[split], (split, len(rows))
        assert {int(r['label_id']) for r in rows} == set(range(NUM_CLASSES))
        for r in rows:
            rid = r['row_id']
            if rid in seen_rows or r['split'] != split or r['original_split'] != split:
                raise ValueError(f'Duplicate row or changed split: {rid}')
            seen_rows.add(rid)
            if int(r['label_id']) not in range(NUM_CLASSES) or r['label'] != CLASSES[int(r['label_id'])]:
                raise ValueError(f'Incorrect class: {rid}')
            for key, mapping in unique_by_key.items():
                v = r[key]
                if not v or (v in mapping and mapping[v] != split):
                    raise ValueError(f'Identity leakage or missing key: {key}, {rid}')
                mapping[v] = split
    assert len(seen_rows) == 7938
    return lock, manifests


def validate_source_shards(repo: pathlib.Path, mirror: pathlib.Path):
    source = json.loads((repo / 'A3-M1/provenance/mirror_downloads.json').read_text())
    shards = {}
    for entry in source['files']:
        relative = entry['path']
        if not relative.endswith('.parquet'):
            continue
        p = mirror / pathlib.Path(relative).relative_to('data/mirror')
        if not p.is_file() or p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
            raise ValueError(f'Original mirror shard absent or SHA-256 mismatch: {p}')
        shards[p.name] = p
    assert len(shards) == 4, 'Exactly four pinned humanitarian Parquet shards required'
    return shards


class Head(nn.Module):
    """Only projections, gate (when learned), and classification heads are trainable."""
    def __init__(self, variant: str, p: float = 0.2):
        super().__init__()
        if variant not in MODELS:
            raise ValueError(variant)
        self.variant = variant
        self.image_proj = nn.Linear(IMAGE_DIM, HIDDEN) if variant != 'text' else None
        self.text_proj = nn.Linear(TEXT_DIM, HIDDEN) if variant != 'image' else None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p)
        if variant in ('image', 'text'):
            self.classifier = nn.Linear(HIDDEN, NUM_CLASSES)
        elif variant == 'concat':
            self.classifier = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(),
                                            nn.Dropout(p), nn.Linear(HIDDEN, NUM_CLASSES))
        else:
            self.gate = nn.Linear(HIDDEN*2, 1) if variant == 'gated' else None
            self.classifier = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                            nn.Dropout(p), nn.Linear(HIDDEN, NUM_CLASSES))

    def forward(self, image: torch.Tensor, text: torch.Tensor, return_gate=False):
        a = self.act(self.image_proj(image)) if self.image_proj is not None else None
        b = self.act(self.text_proj(text)) if self.text_proj is not None else None
        if self.variant == 'image':
            x, gate = self.dropout(a), None
        elif self.variant == 'text':
            x, gate = self.dropout(b), None
        elif self.variant == 'concat':
            x, gate = torch.cat([self.dropout(a), self.dropout(b)], dim=-1), None
        else:
            gate = torch.sigmoid(self.gate(torch.cat([a,b],dim=-1))) if self.gate is not None else torch.full_like(a[:, :1], .5)
            x = self.dropout(gate*a + (1.0-gate)*b)
        logits = self.classifier(x)
        return (logits, gate) if return_gate else logits


def load_features(path: pathlib.Path, *, expected_split: str | None = None):
    with np.load(path, allow_pickle=False) as f:
        needed = {'row_ids', 'group_ids', 'y', 'image', 'text'}
        assert needed.issubset(set(f.files)), f'{path}: missing fields'
        data = {k: f[k].copy() for k in needed}
    n = len(data['y'])
    assert all(len(v) == n for v in data.values())
    assert data['image'].shape == (n, IMAGE_DIM)
    assert data['text'].shape == (n, TEXT_DIM)
    assert np.isfinite(data['image']).all() and np.isfinite(data['text']).all()
    assert (data['y'] >= 0).all() and (data['y'] < NUM_CLASSES).all()
    assert len(set(data['row_ids'].tolist())) == n
    if expected_split:
        assert n == COUNTS[expected_split], f'Incorrect {expected_split} count'
        assert all(str(x).startswith(expected_split + ':') for x in data['row_ids'])
    return data


def check_feature_provenance(feat: pathlib.Path, repo: pathlib.Path):
    _, manifests = validate_locked_repository(repo)
    meta = json.loads((feat / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['protocol'] == 'filtered_original'
    for split in SPLITS:
        f = feat / f'{split}.npz'
        if sha256(f) != meta['npz_sha256'][split]:
            raise ValueError(f'Feature file hash mismatch: {f}')
        d = load_features(f, expected_split=split)
        reference = {r['row_id']: (r['group_id'], int(r['label_id'])) for r in manifests[split]}
        actual = {str(r): (str(g), int(y)) for r, g, y in zip(d['row_ids'], d['group_ids'], d['y'])}
        if actual != reference:
            raise ValueError(f'Feature/manifest membership differs: {split}')
    return meta


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(5)), zero_division=0)
    return {
        'accuracy':float(accuracy_score(y_true,y_pred)),
        'macro_f1':float(f1_score(y_true,y_pred,labels=list(range(5)),average='macro',zero_division=0)),
        'per_class': {CLASSES[i]:{'precision':float(precision[i]),'recall':float(recall[i]),
                                  'f1':float(f1[i]),'support':int(support[i])} for i in range(5)},
        'confusion_matrix':confusion_matrix(y_true,y_pred,labels=list(range(5))).tolist(),
    }


def predict(model: Head, data: dict, device: torch.device, batch_size=512):
    model.eval(); y_pred=[]; all_gate=[]; start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(data['y']),batch_size):
            a=torch.from_numpy(data['image'][i:i+batch_size]).to(device=device,dtype=torch.float32)
            b=torch.from_numpy(data['text'][i:i+batch_size]).to(device=device,dtype=torch.float32)
            logits, gate=model(a,b,return_gate=True)
            y_pred.extend(logits.argmax(-1).cpu().tolist())
            if gate is not None: all_gate.extend(gate.cpu().flatten().tolist())
    return np.array(y_pred,dtype=np.int64), all_gate, time.perf_counter()-start


def train_one(model_name: str, seed: int, feat: pathlib.Path, runs: pathlib.Path,
              epochs=30, batch_size=64, patience=5, device_name='auto', smoke=False):
    if device_name == 'auto':
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(device_name)
    set_seed(seed)
    data={s:load_features(feat/f'{s}.npz',expected_split=s if not smoke else None) for s in ('train','dev')}
    if not smoke:
        assert len(data['train']['y']) == COUNTS['train'] and len(data['dev']['y']) == COUNTS['dev']
    model=Head(model_name).to(device)
    counts=np.bincount(data['train']['y'],minlength=5)
    assert all(counts), 'Training data must contain all five classes'
    weights=counts.sum()/(5.0*counts.astype(float))
    weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    train_set=TensorDataset(torch.from_numpy(data['train']['image']).float(),
                            torch.from_numpy(data['train']['text']).float(),
                            torch.from_numpy(data['train']['y']).long())
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,
                            generator=torch.Generator().manual_seed(seed),num_workers=0)
    best_f1=-1.; best_loss=float('inf'); best_epoch=0; bad=0
    history=[]; start=time.perf_counter()
    outdir=runs/model_name/f'seed_{seed}'
    outdir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for img,txt,y in train_loader:
            img,txt,y=img.to(device),txt.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(img,txt),y); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval(); dev_logits=[]; dev_loss=[]
        with torch.inference_mode():
            for i in range(0,len(data['dev']['y']),batch_size):
                img=torch.from_numpy(data['dev']['image'][i:i+batch_size]).float().to(device)
                txt=torch.from_numpy(data['dev']['text'][i:i+batch_size]).float().to(device)
                y=torch.from_numpy(data['dev']['y'][i:i+batch_size]).long().to(device)
                logits=model(img,txt); dev_logits.append(logits.argmax(-1).cpu().numpy())
                dev_loss.append((float(criterion(logits,y).cpu()),len(y)))
        preds=np.concatenate(dev_logits)
        m=score(data['dev']['y'],preds)
        loss_dev=sum(loss*n for loss,n in dev_loss)/len(data['dev']['y'])
        entry={'epoch':epoch,'train_loss':float(np.mean(losses)), 'dev_loss':loss_dev,'dev_macro_f1':m['macro_f1']}
        history.append(entry)
        improved = (m['macro_f1'] > best_f1+1e-12 or
                    (abs(m['macro_f1']-best_f1)<=1e-12 and loss_dev < best_loss-1e-12))
        if improved:
            best_f1=m['macro_f1'];best_loss=loss_dev;best_epoch=epoch;bad=0
            torch.save({'state_dict':model.state_dict(),'model':model_name,'seed':seed,
                        'best_epoch':epoch,'dev_macro_f1':best_f1,'dev_loss':best_loss},outdir/'best.pt')
        else: bad+=1
        if bad>=patience: break
    checkpoint=torch.load(outdir/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    preds,gate,infer_time=predict(model,data['dev'],device)
    dev_score=score(data['dev']['y'],preds)
    elapsed=time.perf_counter()-start
    record={'model':model_name,'seed':seed,'best_epoch':best_epoch,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'dev':dev_score,'train_and_dev_seconds':elapsed,'dev_head_inference_seconds':infer_time,
            'device':str(device),'epochs_run':len(history),'smoke_only':smoke,
            'hyperparameters':{'optimizer':'Adam','lr':0.001,'weight_decay':0.0001,
                               'batch_size':batch_size,'dropout':0.2,'max_epochs':epochs,'patience':patience,
                               'class_weights':weights.tolist()},'history':history}
    dump_json(outdir/'run.json',record)
    write_predictions(outdir/'dev_predictions.csv',data['dev'],preds,gate)
    return record


def write_predictions(p: pathlib.Path,data:dict,preds:np.ndarray, gates:list):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['row_id','group_id','y_true','y_pred','gate_image_weight'])
        for i in range(len(preds)):
            w.writerow([str(data['row_ids'][i]),str(data['group_ids'][i]),int(data['y'][i]),
                        int(preds[i]),'' if not gates else f'{gates[i]:.8f}'])


def extract_real_features(repo:pathlib.Path,mirror:pathlib.Path,out:pathlib.Path,
                          batch_size=24,device_name='auto',no_download=False):
    """Stream pinned Parquet images/text. Strict image, text, membership and hash checks."""
    import pyarrow.parquet as pq
    from PIL import Image
    from torchvision import models
    from transformers import AutoModel, AutoTokenizer
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to overwrite existing feature directory: {out}')
    lock, manifests=validate_locked_repository(repo)
    shards=validate_source_shards(repo,mirror)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)
    # HF models/torchvision weights are downloaded by libraries when not cached.
    # Offline use requires their caches; 'no_download' disallows uncached models.
    if no_download:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        if not (pathlib.Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth').exists():
            raise ValueError('ResNet18 pretrained weights not cached for --no-download')
    started=time.perf_counter()
    weights=models.ResNet18_Weights.IMAGENET1K_V1
    image_model=models.resnet18(weights=weights)
    image_model.fc=nn.Identity()
    image_model.eval().to(device)
    transform=weights.transforms()
    text_tokenizer=AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
    text_model=AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION).eval().to(device)
    for p in image_model.parameters():p.requires_grad_(False)
    for p in text_model.parameters():p.requires_grad_(False)
    wanted={s:{r['row_id']:r for r in manifests[s]} for s in SPLITS}
    collected={s:[] for s in SPLITS}; scanned=collections.Counter()
    stats={'token_length':[],'unknown_tokens':0,'truncated':0,'paired_records':0}
    for filename,shard in sorted(shards.items()):
        split='dev' if filename.startswith('validation-') else filename.split('-')[0]
        assert split in SPLITS
        pending=[]
        def flush():
            if not pending:return
            imgs=[];texts=[]
            for r,item in pending:
                blob=(item.get('image') or {}).get('bytes')
                raw_text=item.get('text') or ''
                if not blob or not raw_text:
                    raise ValueError(f'Missing modality: {r["row_id"]}')
                if hashlib.sha256(blob).hexdigest()!=r['image_sha256']:
                    raise ValueError(f'Image hash mismatch: {r["row_id"]}')
                if hashlib.sha256(raw_text.encode()).hexdigest()!=r['raw_text_sha256']:
                    raise ValueError(f'Text hash mismatch: {r["row_id"]}')
                norm=normalise_text(raw_text)
                if hashlib.sha256(norm.encode()).hexdigest()!=r['normalized_text_sha256']:
                    raise ValueError(f'Normalization drift: {r["row_id"]}')
                im=Image.open(io.BytesIO(blob)); im.load()
                imgs.append(transform(im.convert('RGB')))
                texts.append(norm)
            with torch.inference_mode():
                image_features=image_model(torch.stack(imgs).to(device)).cpu().numpy().astype('float32')
                tokens=text_tokenizer(texts,max_length=MAX_TOKENS,padding='max_length',truncation=True,return_tensors='pt')
                ids=tokens['input_ids'].to(device);mask=tokens['attention_mask'].to(device)
                hidden=text_model(input_ids=ids,attention_mask=mask).last_hidden_state
                # DistilBERT has no pooler: attention-mask-weighted mean.
                m=mask.unsqueeze(-1).to(hidden.dtype)
                text_features=((hidden*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy().astype('float32')
            for i,(r,item) in enumerate(pending):
                if int(item['label']) != int(r['label_id']) or str(item['tweet_id']) != r['tweet_id'] or item['event_name'] != r['event_name']:
                    raise ValueError(f'Metadata/label mismatch: {r["row_id"]}')
                token_ids=text_tokenizer(texts[i],truncation=False,add_special_tokens=True)['input_ids']
                stats['token_length'].append(len(token_ids))
                stats['truncated']+=int(len(token_ids)>MAX_TOKENS)
                stats['unknown_tokens']+=sum(x==text_tokenizer.unk_token_id for x in token_ids)
                collected[split].append((r['row_id'],r['group_id'],int(r['label_id']),image_features[i],text_features[i]))
            pending.clear()
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=48):
            for item in batch.to_pylist():
                rowid=f"{split}:{item['image_id']}"
                scanned[split]+=1
                if rowid not in wanted[split]:continue
                pending.append((wanted[split].pop(rowid),item))
                if len(pending)>=batch_size:flush()
        flush()
    if any(wanted[s] for s in SPLITS):
        raise ValueError('Missing locked rows in mirror: '+str({s:len(v) for s,v in wanted.items()}))
    if dict(scanned)!={'train':6126,'dev':998,'test':955}:
        raise ValueError(f'Mirror population changed: {scanned}')
    out.mkdir(parents=True,exist_ok=True)
    fingerprints={}
    for split in SPLITS:
        rows=sorted(collected[split],key=lambda x:x[0])
        assert len(rows)==COUNTS[split], (split,len(rows))
        fn=out/f'{split}.npz'
        np.savez_compressed(fn,row_ids=np.array([x[0] for x in rows]),
                            group_ids=np.array([x[1] for x in rows]),
                            y=np.array([x[2] for x in rows],dtype=np.int64),
                            image=np.stack([x[3] for x in rows]),
                            text=np.stack([x[4] for x in rows]))
        fingerprints[split]=sha256(fn)
    duration=time.perf_counter()-started
    dump_json(out/'metadata.json',{'protocol':'filtered_original','dataset_version':lock['dataset_version'],
        'npz_sha256':fingerprints,'mirror_revision':'a935fda3cee8a5250746a8d1d6278214d615cc89',
        'pretrained_image':'ResNet18 IMAGENET1K_V1',
        'pretrained_text':TEXT_MODEL,'text_model_commit':getattr(text_model.config,'_commit_hash',None),
        'tokenizer_model':TEXT_MODEL,'model_revision_locked':TEXT_REVISION,'max_tokens':MAX_TOKENS,'mean_token_length':float(np.mean(stats['token_length'])),
        'max_token_length':int(max(stats['token_length'])), 'truncated_pairs':stats['truncated'],
        'unknown_tokens':stats['unknown_tokens'],'encoder_feature_seconds':duration,
        'device':str(device),'python':platform.python_version(),'torch':torch.__version__,
        'torchvision':__import__('torchvision').__version__,
        'transformers':__import__('transformers').__version__,
        'pyarrow':pq.__version__ if hasattr(pq,'__version__') else __import__('pyarrow').__version__})
    check_feature_provenance(out,repo)
    return out


def evaluate_final(repo:pathlib.Path,feat:pathlib.Path,runs:pathlib.Path,out:pathlib.Path,
                   batch_size=512,device_name='auto',bootstrap=500):
    """No configuration or seed selection based on test. Requires all 15 dev-selected checkpoints."""
    from sklearn.metrics import f1_score
    meta=check_feature_provenance(feat,repo)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite final results')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)
    data=load_features(feat/'test.npz',expected_split='test')
    all_pred={}; metrics=[]
    for variant in MODELS:
        for seed in SEEDS:
            d=runs/variant/f'seed_{seed}'
            log=json.loads((d/'run.json').read_text())
            if log['smoke_only'] or log['model']!=variant or log['seed']!=seed:
                raise ValueError(f'Invalid or synthetic checkpoint: {d}')
            ck=torch.load(d/'best.pt',map_location=device,weights_only=True)
            if ck['model']!=variant or ck['seed']!=seed or ck['best_epoch']!=log['best_epoch']:
                raise ValueError(f'Checkpoint metadata differs: {d}')
            model=Head(variant).to(device);model.load_state_dict(ck['state_dict'])
            pred,gates,duration=predict(model,data,device,batch_size=batch_size)
            m=score(data['y'],pred)
            all_pred[variant,seed]=pred
            metrics.append({'model':variant,'seed':seed,'macro_f1':m['macro_f1'],
                            'accuracy':m['accuracy'],'test_head_inference_seconds':duration,
                            'best_epoch':log['best_epoch'],'train_and_dev_seconds':log['train_and_dev_seconds'],
                            'trainable_parameters':log['trainable_parameters']})
            path=out/variant/f'seed_{seed}'
            path.mkdir(parents=True,exist_ok=True)
            dump_json(path/'test_metrics.json',m)
            write_predictions(path/'test_predictions.csv',data,pred,gates)
    with (out/'per_seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    summary=[]
    for variant in MODELS:
        subset=[m for m in metrics if m['model']==variant]
        summary.append({'model':variant,'macro_f1_mean':statistics.mean(x['macro_f1'] for x in subset),
                        'macro_f1_sd':statistics.stdev(x['macro_f1'] for x in subset),
                        'accuracy_mean':statistics.mean(x['accuracy'] for x in subset),
                        'accuracy_sd':statistics.stdev(x['accuracy'] for x in subset)})
    dump_json(out/'summary.json',{'dataset_version':meta['dataset_version'],'test_protocol':'custom_filtered_original',
                                  'seeds':list(SEEDS),'model_metrics':summary,
                                  'warning':'Custom filtered test; rare class has 9 examples; not an untouched official benchmark.'})
    # Connected-group bootstrap for paired macro-F1 differences; no post-hoc tuning.
    ids=collections.defaultdict(list)
    for i,g in enumerate(data['group_ids']):ids[str(g)].append(i)
    groups=list(ids.values())
    rng=np.random.default_rng(20261117)
    bootrows=[]
    for seed in SEEDS:
        gate=all_pred['gated',seed];base=all_pred['concat',seed]
        observed=score(data['y'],gate)['macro_f1']-score(data['y'],base)['macro_f1']
        diffs=[]
        for _ in range(bootstrap):
            selection=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
            a=f1_score(data['y'][selection],gate[selection],labels=list(range(5)),average='macro',zero_division=0)
            b=f1_score(data['y'][selection],base[selection],labels=list(range(5)),average='macro',zero_division=0)
            diffs.append(float(a-b))
        bootrows.append({'seed':seed,'observed_delta_macro_f1':observed,
                         'group_bootstrap_ci95_low':float(np.quantile(diffs,.025)),
                         'group_bootstrap_ci95_high':float(np.quantile(diffs,.975))})
    dump_json(out/'paired_group_bootstrap.json',{'comparison':'gated_minus_concat',
        'clusters':len(groups),'replicates_per_seed':bootstrap,'rows':bootrows,
        'caution':'Descriptive CIs on fixed custom test; multiple seeds share the same sample groups.'})
    # Operational categories are prediction disagreement patterns, NOT verified semantic conflict.
    with (out/'modality_cases.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','group_id','seed','true_label',
                                   'image_pred','text_pred','concat_pred','gated_pred',
                                   'both_unimodal_correct','unimodal_disagree','fusion_recovers_one_correct',
                                   'gated_harms_one_correct'])
        for seed in SEEDS:
            im=all_pred['image',seed];tx=all_pred['text',seed]
            co=all_pred['concat',seed];ga=all_pred['gated',seed]
            for i,y in enumerate(data['y']):
                any_correct=(im[i]==y or tx[i]==y)
                w.writerow([data['row_ids'][i],data['group_ids'][i],seed,int(y),int(im[i]),int(tx[i]),
                            int(co[i]),int(ga[i]),int(im[i]==y and tx[i]==y),int(im[i]!=tx[i]),
                            int(any_correct and ga[i]==y),int(any_correct and ga[i]!=y)])
    return out


def run_smoke(out:pathlib.Path):
    """Isolated synthetic check ONLY; never a model score for the assignment."""
    if out.exists() and any(out.iterdir()):raise ValueError('Smoke output must be empty')
    feat=out/'features'; feat.mkdir(parents=True)
    rng=np.random.default_rng(9090)
    for split,n in [('train',75),('dev',25)]:
        y=np.arange(n)%5
        np.savez_compressed(feat/f'{split}.npz',image=rng.standard_normal((n,512)).astype('float32'),
                            text=rng.standard_normal((n,768)).astype('float32'),y=y.astype('int64'),
                            row_ids=np.array([f'{split}:synthetic_{i}' for i in range(n)]),
                            group_ids=np.array([f'syn_{split}_{i}' for i in range(n)]))
    results=[]
    for model in MODELS:
        r=train_one(model,42,feat,out/'runs',epochs=2,batch_size=16,patience=2,smoke=True,device_name='cpu')
        results.append({'model':model,'shapes_and_training':'PASS','trained_on':'SYNTHETIC ONLY',
                        'parameters':r['trainable_parameters']})
    dump_json(out/'SMOKE_ONLY.json',{'status':'PASS','cannot_be_used_as_assignment_metrics':True,'tests':results})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('check',help='Verify published M1 manifests and SHA-256 locks')
    s.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    s=sub.add_parser('extract',help='Extract pretrained features from actual pinned mirror Parquet')
    s.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    s.add_argument('--mirror',type=pathlib.Path,required=True)
    s.add_argument('--out',type=pathlib.Path,required=True)
    s.add_argument('--batch-size',type=int,default=24)
    s.add_argument('--device',default='auto');s.add_argument('--no-download',action='store_true')
    s=sub.add_parser('train',help='Train one variant/seed on real cached features and dev only')
    s.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    s.add_argument('--features',type=pathlib.Path,required=True)
    s.add_argument('--runs',type=pathlib.Path,required=True)
    s.add_argument('--model',choices=MODELS,required=True);s.add_argument('--seed',type=int,choices=SEEDS,required=True)
    s.add_argument('--epochs',type=int,default=30);s.add_argument('--batch-size',type=int,default=64)
    s.add_argument('--device',default='auto')
    s=sub.add_parser('finalize',help='Run once, AFTER all 15 dev-selected training runs')
    s.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    s.add_argument('--features',type=pathlib.Path,required=True)
    s.add_argument('--runs',type=pathlib.Path,required=True)
    s.add_argument('--out',type=pathlib.Path,required=True)
    s.add_argument('--bootstrap',type=int,default=500)
    s.add_argument('--device',default='auto')
    s=sub.add_parser('smoke',help='Fast isolated synthetic test (NOT assignment metrics)')
    s.add_argument('--out',type=pathlib.Path,required=True)
    args=p.parse_args()
    if args.cmd=='check':
        lock,rows=validate_locked_repository(args.repo)
        print(json.dumps({'status':'PASS','protocol':lock['primary_protocol'],
                          'split_counts':{s:len(v) for s,v in rows.items()}},indent=2))
    elif args.cmd=='extract':
        extract_real_features(args.repo,args.mirror,args.out,args.batch_size,args.device,args.no_download)
        print('REAL FEATURE EXTRACTION PASS:',args.out)
    elif args.cmd=='train':
        check_feature_provenance(args.features,args.repo)
        out=args.runs/args.model/f'seed_{args.seed}'
        if out.exists() and any(out.iterdir()):raise ValueError(f'Refusing to overwrite run: {out}')
        r=train_one(args.model,args.seed,args.features,args.runs,args.epochs,args.batch_size,device_name=args.device)
        print(json.dumps({'model':r['model'],'seed':r['seed'],'dev_macro_f1':r['dev']['macro_f1'],
                          'best_epoch':r['best_epoch']},indent=2))
    elif args.cmd=='finalize':
        evaluate_final(args.repo,args.features,args.runs,args.out,device_name=args.device,bootstrap=args.bootstrap)
        print('REAL TEST EVALUATION COMPLETE:',args.out)
    elif args.cmd=='smoke':
        print(json.dumps(run_smoke(args.out),indent=2))

if __name__=='__main__':main()
