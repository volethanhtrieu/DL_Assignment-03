#!/usr/bin/env python3
"""Stratified byte-level check of mirror images against pinned official QCRI."""
import sys, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime-deps'))
import pyarrow.parquet as pq
import pandas as pd
import csv,json,hashlib,concurrent.futures as cf
from fetch_data import get,BASE

if __name__=='__main__':
    lookup={}
    for path in (ROOT/'data/annotations').glob('*_final_data.tsv'):
        with path.open() as f:
            for row in csv.DictReader(f,delimiter='\t'):
                lookup[(path.stem.replace('_final_data',''),row['image_id'])]=row['image_path']
    rows=[]
    for path in sorted((ROOT/'data/mirror/humanitarian').glob('*.parquet')):
        frame=pq.read_table(path,columns=['image_id','event_name','label']).to_pandas()
        frame['split']=path.name.split('-')[0].replace('validation','dev')
        rows.extend(frame.to_dict('records'))
    data=pd.DataFrame(rows)
    data['rank']=data.image_id.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    selected=data.sort_values('rank').groupby(['split','event_name']).head(2)
    primary=[]
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        tasks={pool.submit(get,BASE+lookup[(r.event_name,r.image_id)],ROOT/'data/images'/lookup[(r.event_name,r.image_id)]):r for r in selected.itertuples()}
        for f in cf.as_completed(tasks):
            r=tasks[f];result=f.result();result.update({'split':r.split,'event_name':r.event_name,'image_id':r.image_id})
            primary.append(result)
    (ROOT/'provenance/stratified_primary_downloads.json').write_text(json.dumps({'rule':'two lowest SHA256(image_id) per split and event','files':primary},indent=2))
    print(json.dumps({'sampled':len(primary),'successful':sum(x['status']!='failed' for x in primary),'failed':[x for x in primary if x['status']=='failed']}),flush=True)
