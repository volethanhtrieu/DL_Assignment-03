#!/usr/bin/env python3
"""Fallback mirror, explicitly third-party. Primary QCRI annotations remain authority."""
import concurrent.futures as cf, json
from fetch_data import ROOT, get

MIRROR_REV='a935fda3cee8a5250746a8d1d6278214d615cc89'
MIRROR='https://huggingface.co/datasets/pranitchawla/crisismmd-mteb/resolve/'+MIRROR_REV+'/'
PATHS=['README.md','humanitarian/test-00000-of-00001.parquet','humanitarian/validation-00000-of-00001.parquet',
       'humanitarian/train-00000-of-00002.parquet','humanitarian/train-00001-of-00002.parquet']

if __name__=='__main__':
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(get,MIRROR+p,ROOT/'data/mirror'/p):p for p in PATHS}
        results=[]
        for future in cf.as_completed(jobs):
            r=future.result();results.append(r)
            print(json.dumps({k:v for k,v in r.items() if k!='url'}),flush=True)
            (ROOT/'provenance/mirror_downloads.json').write_text(json.dumps({'mirror_revision':MIRROR_REV,'qcri_revision':'10a5626ba112ab9c50369c473a2062bcf205d708','files':results},indent=2))
