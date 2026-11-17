#!/usr/bin/env python3
"""Execute fixed M1 plan: 3 M2 models + gated and fixed-gate M3, 3 seeds each.
This does not silently run final held-out test; `finalize` is a distinct human action.
"""
import argparse
import pathlib
import subprocess
import sys

MODELS=['image','text','concat','gated','fixed_gate']
SEEDS=[42,43,44]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--features',type=pathlib.Path,required=True)
    p.add_argument('--runs',type=pathlib.Path,required=True)
    p.add_argument('--milestone',choices=['m2','all'],default='m2')
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    script=pathlib.Path(__file__).with_name('experiment.py')
    for model in MODELS[:3] if a.milestone=='m2' else MODELS:
        for seed in SEEDS:
            expected=a.runs/model/f'seed_{seed}'/'run.json'
            if expected.exists():
                raise SystemExit(f'Existing run found: {expected}; inspect before rerunning (no overwrite).')
            command=[sys.executable,str(script),'train','--repo',str(a.repo),
                    '--features',str(a.features),'--runs',str(a.runs),
                    '--model',model,'--seed',str(seed),'--device',a.device]
            print('EXECUTING:', ' '.join(command),flush=True)
            subprocess.run(command,check=True)
    print('Finished development-selected model training. Do NOT use test to tune.')
