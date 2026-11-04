#!/usr/bin/env python3
"""Download pinned QCRI metadata/images for auditing only; never trains models."""
import argparse, concurrent.futures as cf, hashlib, json, pathlib, time, urllib.request, urllib.error

REVISION = '10a5626ba112ab9c50369c473a2062bcf205d708'
BASE = 'https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/' + REVISION + '/'
ROOT = pathlib.Path(__file__).resolve().parents[1]

def get(url, target, attempts=3):
    target = pathlib.Path(target)
    if target.exists() and target.stat().st_size:
        return {'url':url,'path':str(target.relative_to(ROOT)), 'status':'cached', 'bytes':target.stat().st_size,
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True,exist_ok=True)
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={'User-Agent':'CO3133-A3-M1-dataset-audit/1.0'})
            with urllib.request.urlopen(request,timeout=35) as response:
                data=response.read()
                if response.status != 200: raise RuntimeError('HTTP '+str(response.status))
            tmp=target.with_suffix(target.suffix+'.part')
            tmp.write_bytes(data); tmp.replace(target)
            return {'url':url,'path':str(target.relative_to(ROOT)), 'status':'downloaded','bytes':len(data),
                    'sha256':hashlib.sha256(data).hexdigest()}
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            last=f'{type(exc).__name__}: {exc}'
            if isinstance(exc,urllib.error.HTTPError) and exc.code in (401,403,404): break
            if attempt+1<attempts: time.sleep(min(2**attempt,4))
    return {'url':url,'path':str(target.relative_to(ROOT)), 'status':'failed','error':last}

def metadata():
    items=[('README.md',ROOT/'provenance/QCRI_README.md'),
           ('licenses_by-nc-sa_4.0_legalcode.txt',ROOT/'provenance/licenses_by-nc-sa_4.0_legalcode.txt')]
    items += [(f'humanitarian/{split}.json', ROOT/f'data/annotations/humanitarian_{split}.json') for split in ['train','dev','test']]
    events=['california_wildfires','hurricane_harvey','hurricane_irma','hurricane_maria',
            'iraq_iran_earthquake','mexico_earthquake','srilanka_floods']
    items += [(f'files_individual_events/{event}_final_data.tsv', ROOT/f'data/annotations/{event}_final_data.tsv') for event in events]
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda x:get(BASE+x[0],x[1]),items))
    archive='https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip'
    results.append(get(archive,ROOT/'data/annotations/crisismmd_datasplit_agreed_label.zip',attempts=1))
    (ROOT/'provenance/metadata_downloads.json').write_text(json.dumps({'revision':REVISION,'files':results},indent=2))
    print(json.dumps({'metadata_ok':sum(x['status']!='failed' for x in results), 'metadata_failed':[x for x in results if x['status']=='failed']}),flush=True)

def images(manifest,workers,limit):
    import csv
    with open(manifest,newline='') as handle: rows=list(csv.DictReader(handle))
    paths=list(dict.fromkeys(row['source_image_path'] for row in rows))
    if limit: paths=paths[:limit]
    log=ROOT/'provenance/image_downloads.jsonl'
    start=time.monotonic(); failed=0; total_bytes=0
    with log.open('w') as stream, cf.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(get,BASE+path,ROOT/'data/images'/path):path for path in paths}
        for n,future in enumerate(cf.as_completed(jobs),1):
            result=future.result(); result['source_path']=jobs[future]
            stream.write(json.dumps(result)+'\n'); stream.flush()
            failed+=result['status']=='failed'; total_bytes+=result.get('bytes',0)
            if n%250==0 or n==len(paths):
                print(json.dumps({'completed':n,'target':len(paths),'failed':failed,'MB':round(total_bytes/1e6,1),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','images'])
    p.add_argument('--manifest',default=str(ROOT/'tables/benchmark_manifest.csv'));p.add_argument('--workers',type=int,default=12)
    p.add_argument('--limit',type=int,default=0); args=p.parse_args()
    if args.stage=='metadata':metadata()
    else:images(args.manifest,args.workers,args.limit)
