#!/usr/bin/env python3
"""Download only the exact M1-pinned CrisisMMD mirror Parquet shards.

CrisisMMD CC BY-NC-SA 4.0. Never commit these shards to a public repository.
"""
import argparse
import hashlib
import json
import os
import pathlib
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def fetch(repo, target):
    lock=json.loads((repo/'A3-M1/provenance/mirror_downloads.json').read_text())
    assert lock['mirror_revision']=='a935fda3cee8a5250746a8d1d6278214d615cc89'
    for x in lock['files']:
        if not x['path'].endswith('.parquet'):continue
        path=target/pathlib.Path(x['path']).relative_to('data/mirror')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==x['bytes'] and digest(path)==x['sha256']:
            print('EXISTS/VERIFIED',path,flush=True);continue
        part=path.with_suffix(path.suffix+'.part')
        if part.exists():part.unlink()
        print('DOWNLOADING',x['url'],'->',path,flush=True)
        try:
            req=urllib.request.Request(x['url'],headers={'User-Agent':'CO3133-A3-Research/1.0'})
            with urllib.request.urlopen(req,timeout=90) as inp,part.open('wb') as out:
                while True:
                    b=inp.read(1024*1024)
                    if not b:break
                    out.write(b)
            if part.stat().st_size!=x['bytes'] or digest(part)!=x['sha256']:
                raise ValueError('Size or SHA256 differs from the M1 lock: '+str(path))
            os.replace(part,path)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    print('All FOUR pinned humanitarian shards verified. Raw data stays local.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();fetch(a.repo,a.out)
