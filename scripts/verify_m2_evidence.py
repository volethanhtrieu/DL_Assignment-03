#!/usr/bin/env python3
"""Independently verify the published REAL M2 development predictions (standard library only).

Run at the root of DL_Assignment-03 after applying the M2 publication patch.
The purpose is evidence-integrity verification, not retraining or held-out test evaluation.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'artifacts/m2/experiments/primary'
MODELS=('image','text','concat')
SEEDS=(42,43,44)
LABEL_SUPPORT=(9,80,148,235,514)
DATASET_VERSION='crisismmd5-mirror-remediation-20261104-v1'

def require(condition,message):
    if not condition: raise AssertionError(message)

def parse_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def metrics(rows):
    confusion=[[0]*5 for _ in range(5)]
    for row in rows:
        t=int(row['y_true']); p=int(row['y_pred'])
        require(t in range(5) and p in range(5), 'Unexpected label ID')
        confusion[t][p]+=1
    acc=sum(confusion[i][i] for i in range(5))/len(rows)
    f1=[]
    for i in range(5):
        tp=confusion[i][i]
        fp=sum(confusion[j][i] for j in range(5))-tp
        fn=sum(confusion[i][j] for j in range(5))-tp
        f1.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)
    return acc,sum(f1)/5,confusion

def check_hashes():
    manifest=ROOT/'docs/m2/SHA256SUMS.txt'
    checks=[]
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        expected,relative=line.split('  ',1)
        path=ROOT/relative
        require(path.is_file(),'Missing checksum file: '+relative)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual==expected,'SHA256 mismatch: '+relative)
        checks.append(relative)
    require(len(checks)>=35,'Unexpectedly short checksum ledger')
    return len(checks)

def main():
    hashes=check_hashes()
    lock=json.loads((ROOT/'A3-M1/DATASET_LOCK.json').read_text())
    require(lock['dataset_version']==DATASET_VERSION, 'M1 dataset version changed')
    require(lock['primary_protocol']=='filtered_original', 'M1 split protocol changed')
    require(lock['split_counts']=={'train':6001,'dev':986,'test':951}, 'M1 split counts changed')
    md=json.loads((ROOT/'artifacts/m2/feature_metadata.json').read_text())
    require(md['dataset_version']==DATASET_VERSION and md['protocol']=='filtered_original','Feature metadata mismatch')
    require(md['mirror_revision']=='a935fda3cee8a5250746a8d1d6278214d615cc89','Mirror revision mismatch')
    ref_ids=None; ref_truth=None; ref_groups=None
    by_model={}
    for model in MODELS:
        results=[]
        for seed in SEEDS:
            directory=EVIDENCE/model/('seed_'+str(seed))
            run=json.loads((directory/'run.json').read_text())
            require(run['model']==model and run['seed']==seed and run['smoke_only'] is False,'Wrong run identity or smoke data')
            require((directory/'best.pt').stat().st_size>1000,'Missing/empty checkpoint')
            rows=parse_csv(directory/'dev_predictions.csv')
            require(len(rows)==986 and len({r['row_id'] for r in rows})==986,'Invalid dev set size or duplicate IDs')
            current_ids=tuple(r['row_id'] for r in rows)
            current_true=tuple(int(r['y_true']) for r in rows)
            current_groups=tuple(r['group_id'] for r in rows)
            if ref_ids is None: ref_ids,ref_truth,ref_groups=current_ids,current_true,current_groups
            else: require(current_ids==ref_ids and current_true==ref_truth and current_groups==ref_groups,'Dev alignment differs between runs')
            acc,macro,conf=metrics(rows)
            require([sum(row) for row in conf]==list(LABEL_SUPPORT),'Rare class/support mismatch')
            require(math.isclose(acc,run['dev']['accuracy'],rel_tol=0,abs_tol=1e-12),'Accuracy mismatch')
            require(math.isclose(macro,run['dev']['macro_f1'],rel_tol=0,abs_tol=1e-12),'Macro-F1 mismatch')
            require(conf==run['dev']['confusion_matrix'],'Confusion mismatch')
            require(any(h['epoch']==run['best_epoch'] and math.isclose(h['dev_macro_f1'],macro,abs_tol=1e-12) for h in run['history']),'Best epoch/history mismatch')
            results.append((seed,macro,acc,run['trainable_parameters']))
        by_model[model]=results
    summary={r['model']:r for r in parse_csv(ROOT/'results/m2_dev/summary_dev.csv')}
    individual={(r['model'],int(r['seed'])):r for r in parse_csv(ROOT/'results/m2_dev/individual_dev.csv')}
    require(set(summary)==set(MODELS) and len(individual)==9,'Incomplete M2 summary tables')
    for model,values in by_model.items():
        a=summary[model]; f1=[v[1] for v in values]; acc=[v[2] for v in values]
        for k,v in [('dev_macro_f1_mean',statistics.mean(f1)),('dev_macro_f1_sd',statistics.stdev(f1)),('dev_accuracy_mean',statistics.mean(acc)),('dev_accuracy_sd',statistics.stdev(acc))]:
            require(math.isclose(float(a[k]),v,rel_tol=0,abs_tol=1e-12),model+' '+k+' mismatch')
        for seed,f,ac,params in values:
            item=individual[model,seed]
            require(math.isclose(float(item['dev_macro_f1']),f,abs_tol=1e-12),'Individual F1 mismatch')
            require(math.isclose(float(item['dev_accuracy']),ac,abs_tol=1e-12),'Individual accuracy mismatch')
            require(params==int(item['trainable_parameters'])==int(a['trainable_parameters']),'Parameter count mismatch')
    print(json.dumps({'status':'PASS','sha256_files_checked':hashes,'real_dev_runs':9,'dev_predictions_total':9*986,'no_test_metrics':True,'models':{m:{'macro_f1_mean':statistics.mean(v[1] for v in xs),'accuracy_mean':statistics.mean(v[2] for v in xs)} for m,xs in by_model.items()}},indent=2))

if __name__=='__main__':main()
