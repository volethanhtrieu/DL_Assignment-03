#!/usr/bin/env python3
"""Development-only quantitative tables for M2/M3; NEVER reads test or selects best seed."""
import argparse
import csv
import json
import pathlib
import statistics

MODELS=('image','text','concat','gated','fixed_gate')
SEEDS=(42,43,44)


def summarize(runs, out, milestone):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite development summary')
    out.mkdir(parents=True,exist_ok=True)
    names=MODELS[:3] if milestone=='m2' else MODELS
    individual=[]
    totals=[]
    for variant in names:
        records=[]
        for seed in SEEDS:
            p=runs/variant/f'seed_{seed}'/'run.json'
            log=json.loads(p.read_text())
            if log['model']!=variant or log['seed']!=seed or log['smoke_only']:
                raise ValueError(f'Synthetic/wrong run: {p}')
            if not (p.parent/'best.pt').exists() or not (p.parent/'dev_predictions.csv').exists():
                raise ValueError(f'Incomplete run: {p.parent}')
            individual.append({'model':variant,'seed':seed,'dev_macro_f1':log['dev']['macro_f1'],
                               'dev_accuracy':log['dev']['accuracy'],
                               'train_seconds':log['train_and_dev_seconds'],
                               'best_epoch':log['best_epoch'],
                               'trainable_parameters':log['trainable_parameters']})
            records.append(log)
        totals.append({'model':variant,
          'dev_macro_f1_mean':statistics.mean(x['dev']['macro_f1'] for x in records),
          'dev_macro_f1_sd':statistics.stdev(x['dev']['macro_f1'] for x in records),
          'dev_accuracy_mean':statistics.mean(x['dev']['accuracy'] for x in records),
          'dev_accuracy_sd':statistics.stdev(x['dev']['accuracy'] for x in records),
          'train_seconds_sum':sum(x['train_and_dev_seconds'] for x in records),
          'trainable_parameters':records[0]['trainable_parameters']})
    for path,rows in [(out/'individual_dev.csv',individual),(out/'summary_dev.csv',totals)]:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Real dev metrics: '+milestone.upper(),'',
           '> Generated strictly from REAL model checkpoints/run.json, not from synthetic smoke or test.', '',
           '| Model | Dev macro-F1 mean ± SD | Dev accuracy mean ± SD | Trainable params |',
           '|---|---:|---:|---:|']
    for x in totals:
        lines.append(f"| {x['model']} | {x['dev_macro_f1_mean']:.4f} ± {x['dev_macro_f1_sd']:.4f} | {x['dev_accuracy_mean']:.4f} ± {x['dev_accuracy_sd']:.4f} | {x['trainable_parameters']:,} |")
    lines+=['','**Limits:** dev-only; group-multimodal leakage controlled by locked manifests; rare class has only 9 dev examples. Do not make out-of-event generalization claims.','']
    (out/'DEV_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return totals


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--milestone',choices=['m2','all'],default='m2')
    a=ap.parse_args();summarize(a.runs,a.out,a.milestone)
    print('Development-only tables:',a.out)
