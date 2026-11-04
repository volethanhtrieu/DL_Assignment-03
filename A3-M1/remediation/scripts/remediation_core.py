#!/usr/bin/env python3
"""Shared cached-input and grouping helpers. Never writes original A3-M1 inputs."""
import pathlib,sys,json,collections,hashlib
HERE=pathlib.Path(__file__).resolve().parents[1]
BASE=HERE.parent
sys.path.insert(0,str(BASE/'runtime-deps'))
import pandas as pd

KEYS=['tweet_id','image_sha256','pixel_sha256','raw_text_sha256','normalized_text_sha256']
JOIN_DECISIONS={'duplicate','duplicate_asset_variant','uncertain_same_scene'}

class UnionFind:
    def __init__(self,ids): self.parent={x:x for x in ids}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def groups(self):
        out=collections.defaultdict(list)
        for x in sorted(self.parent):out[self.find(x)].append(x)
        return list(out.values())

def load_inputs():
    df=pd.read_csv(BASE/'tables/benchmark_manifest.csv',dtype={'tweet_id':str,'image_id':str,'phash64':str}).fillna('')
    original=pd.read_csv(HERE/'review/candidate_index.csv',dtype=str)
    decision={}
    for item in json.loads((HERE/'review/decisions_original.json').read_text()):
        for i in item['ids']:
            key=f'P{i:03}'
            assert key not in decision
            decision[key]={'decision':item['decision'],'note':item['note']}
    assert set(decision)==set(original.review_id)
    for field in ['decision','note']:original[field]=original.review_id.map(lambda x:decision[x][field])
    return df,original

def input_groups(df,reviewed):
    uf=UnionFind(df.row_id)
    for key in KEYS:
        for value,part in df.groupby(key,sort=True):
            if value=='':continue
            ids=part.row_id.tolist()
            for other in ids[1:]:uf.union(ids[0],other)
    for r in reviewed.itertuples():
        if r.decision in JOIN_DECISIONS:uf.union(r.row_a,r.row_b)
    return uf

def pair_key(a,b):return tuple(sorted([a,b]))

def placeholder_rows(df,original):
    # Only visually confirmed missing-image assets and byte/pixel-identical copies.
    r=original[original.decision=='placeholder_asset']
    seeds=set(r.image_a)|set(r.image_b)
    seedrows=df[df.image_id.isin(seeds)]
    same=df.image_sha256.isin(seedrows.image_sha256)|df.pixel_sha256.isin(seedrows.pixel_sha256)
    return set(df.loc[same,'row_id']),seeds

def sha_file(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
