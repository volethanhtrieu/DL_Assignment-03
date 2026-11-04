#!/usr/bin/env python3
"""Replay the locked primary split without downloads or training.

Default: replay membership from published, previously audited group metadata.
--audited-manifest: also rebuild groups/complete CSVs from an existing full
source-audit manifest. This does not decode images or recompute content hashes.
"""
import argparse,collections,csv,hashlib,io,json,pathlib

ROOT=pathlib.Path(__file__).resolve().parents[1]
M1=ROOT/'A3-M1'
REM=M1/'remediation'
PRIORITY={'train':0,'dev':1,'test':2}
KEYS=['tweet_id','image_sha256','pixel_sha256','raw_text_sha256','normalized_text_sha256']

def rows(path):
    with pathlib.Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def digest(data):return hashlib.sha256(data).hexdigest()

def rebuild_groups(raw,review):
    parent={r['row_id']:r['row_id'] for r in raw}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for key in KEYS:
        seen={}
        for r in raw:
            value=r[key]
            if not value:continue
            if value in seen:join(seen[value],r['row_id'])
            else:seen[value]=r['row_id']
    for r in review:
        if r['decision'] in {'duplicate','duplicate_asset_variant'}:join(r['row_a'],r['row_b'])
    groups=collections.defaultdict(list)
    for rid in parent:groups[find(rid)].append(rid)
    ids={}
    for members in groups.values():
        gid='g_'+digest('\n'.join(sorted(members)).encode())[:16]
        ids.update({x:gid for x in members})
    return ids

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=pathlib.Path,required=True,help='New output directory; never overwrites published files.')
    ap.add_argument('--audited-manifest',type=pathlib.Path,help='Optional existing 8,079-row full source audit CSV.')
    args=ap.parse_args();out=args.out.resolve()
    if out==ROOT or ROOT in out.parents:raise SystemExit('Choose an output directory outside this repository.')
    if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be empty.')
    lock=json.loads((M1/'DATASET_LOCK.json').read_text())
    for rel,expected in lock['files'].items():
        actual=digest((ROOT/rel).read_bytes())
        if actual!=expected:raise SystemExit('Published input hash mismatch: '+rel)
    metadata=rows(REM/'tables/group_membership.csv')
    if len(metadata)!=8079 or len({r['row_id'] for r in metadata})!=8079:raise SystemExit('Unexpected source metadata.')
    byid={r['row_id']:r for r in metadata};groups=collections.defaultdict(list)
    for r in metadata:
        if not r['quarantine_reason']:groups[r['base_group_id']].append(r)
    chosen={}
    for gid,g in groups.items():
        winner=max((r['original_split'] for r in g),key=PRIORITY.get)
        for r in g:
            if r['original_split']==winner:chosen[r['row_id']]=winner
    published={};fields=None;class_counts=collections.Counter()
    for split,n in [('train',6001),('dev',986),('test',951)]:
        path=REM/'manifests/filtered_original'/(split+'.csv');data=rows(path)
        if len(data)!=n:raise SystemExit('Split count mismatch: '+split)
        with path.open() as f:fields=next(csv.reader(f))
        for r in data:
            if r['row_id'] in published:raise SystemExit('Duplicate published row ID.')
            if r['split']!=split or r['original_split']!=split:raise SystemExit('Original membership changed.')
            if r['group_id']!=byid[r['row_id']]['base_group_id']:raise SystemExit('Group ID mismatch.')
            published[r['row_id']]=r;class_counts[split,int(r['label_id'])]+=1
        if {int(r['label_id']) for r in data}!=set(range(5)):raise SystemExit('A class is missing.')
    if chosen!={rid:r['split'] for rid,r in published.items()}:raise SystemExit('Membership replay differs from manifests.')
    for key in KEYS+['group_id']:
        seen={}
        for r in published.values():
            if not r[key]:raise SystemExit('Empty required identity key.')
            if r[key] in seen and seen[r[key]]!=r['split']:raise SystemExit('Cross-split '+key)
            seen[r[key]]=r['split']
    expected_counts=rows(REM/'tables/class_counts.csv')
    for r in expected_counts:
        if r['protocol']=='filtered_original' and class_counts[r['split'],int(r['label_id'])]!=int(r['pairs']):raise SystemExit('Class count mismatch.')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'replayed_membership.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['row_id','split','group_id'])
        for rid in sorted(chosen):w.writerow([rid,chosen[rid],byid[rid]['base_group_id']])
    rebuilt=False
    if args.audited_manifest:
        raw=rows(args.audited_manifest)
        if {r['row_id'] for r in raw}!=set(byid):raise SystemExit('Full audit ID set differs.')
        recomputed=rebuild_groups(raw,rows(REM/'review/reviewed_original_134.csv'))
        if any(recomputed[rid]!=r['base_group_id'] for rid,r in byid.items()):raise SystemExit('Group reconstruction differs.')
        source={r['row_id']:r for r in raw}
        for split in PRIORITY:
            selected=[]
            for rid in sorted(r for r,s in chosen.items() if s==split):
                r=dict(source[rid]);r['original_split']=r['split'];r['group_id']=recomputed[rid]
                selected.append({k:r[k] for k in fields})
            stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(selected)
            payload=stream.getvalue().encode();canonical=REM/'manifests/filtered_original'/(split+'.csv')
            if payload!=canonical.read_bytes():raise SystemExit('Rebuilt CSV byte mismatch: '+split)
            (out/(split+'.csv')).write_bytes(payload)
        rebuilt=True
    result={'status':'PASS','source_rows':8079,'quarantined_rows':16,'leakage_filtered_rows':125,'retained_pairs':len(chosen),
        'split_counts':dict(collections.Counter(chosen.values())),'all_five_classes_present':True,
        'identity_checks':'tweet IDs, byte/pixel image hashes, raw/normalized text hashes, connected group IDs',
        'full_csvs_rebuilt_and_byte_identical':rebuilt,
        'scope':'Reuses audited hashes and manual decisions. Does not independently decode data, recompute hashes, re-review photos, train models or establish source archive equivalence.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
