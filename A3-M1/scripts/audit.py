#!/usr/bin/env python3
"""Measure the accessible CrisisMMD humanitarian benchmark; no models or training.

Primary: pinned QCRI annotations. Full paired bytes: explicitly identified MTEB
mirror. The official agreed-label ZIP was inaccessible; equivalence is not assumed.
"""
import sys, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime-deps'))
import csv, json, re, html, unicodedata, hashlib, io, itertools, collections, platform, time, datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
import pyarrow.parquet as pq

LABELS=['affected_individuals','infrastructure_and_utility_damage',
        'rescue_volunteering_or_donation_effort','other_relevant_information','not_humanitarian']
MAP={'injured_or_dead_people':'affected_individuals','missing_or_found_people':'affected_individuals',
     'vehicle_damage':'infrastructure_and_utility_damage'}
PUBLISHED={
 'train':[71,612,912,1279,3252], 'dev':[9,80,149,239,521], 'test':[9,81,126,235,504]}
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):
    s=unicodedata.normalize('NFKC',html.unescape(s)).casefold()
    s=re.sub(r'https?://\S+',' ',s)
    s=re.sub(r'^\s*rt\s+@\w+\s*:\s*','',s)
    s=re.sub(r'@\w+',' ',s)
    return ' '.join(re.findall(r'\w+',s))
def save(name,rows,columns=None):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    frame.to_csv(ROOT/'tables'/name,index=False)
    return frame
def photohash(im):
    a=np.asarray(ImageOps.grayscale(im).resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    low=dctn(a,type=2,norm='ortho')[:8,:8]
    bits=low>np.median(low)
    return f'{sum(int(v)<<i for i,v in enumerate(bits.ravel())):016x}'

def main():
    started=time.monotonic(); raw=[]
    for p in sorted((ROOT/'data/annotations').glob('*_final_data.tsv')):
        with p.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                row['event_name']=p.stem.replace('_final_data',''); raw.append(row)
    indexes=collections.defaultdict(list)
    bypath={}
    for row in raw:
        indexes[(row['event_name'],row['image_id'])].append(row);bypath[row['image_path']]=row
    assert len(raw)==18082,'Unexpected primary source row count; inspect before proceeding'
    confusion=collections.Counter((r['text_human'],r['image_human']) for r in raw)
    save('primary_modality_label_matrix.csv',[{'text_label':a,'image_label':b,'pairs':n} for (a,b),n in sorted(confusion.items())])
    reconstruction=[]; primary_json_issues=[]
    for split in ['train','dev','test']:
        for h in json.loads((ROOT/f'data/annotations/humanitarian_{split}.json').read_text()):
            matches=indexes[(h['event_name'],h['image_id'])]
            src=matches[0] if len(matches)==1 else None
            if not src or src['image_path']!=h['image_path'] or src['tweet_text']!=h['tweet_text'] or h['label']!=src['text_human'] or h['label']!=src['image_human']:
                primary_json_issues.append({'split':split,'image_id':h['image_id'],'tweet_id':str(h['tweet_id']),
                    'json_label':h['label'],'source_text_label':src['text_human'] if src else '',
                    'source_image_label':src['image_human'] if src else '',
                    'missing_id_join':src is None,'path_mismatch':bool(src and src['image_path']!=h['image_path']),
                    'text_mismatch':bool(src and src['tweet_text']!=h['tweet_text']),
                    'image_path':h['image_path']})
            s=bypath.get(h['image_path'])
            if s and s['text_human']==s['image_human']:
                reconstruction.append({'split':split,'image_id':s['image_id'],'source_label':s['text_human'],
                                       'label':MAP.get(s['text_human'],s['text_human'])})
    save('primary_json_annotation_differences.csv',primary_json_issues)
    save('reconstructed_counts.csv',pd.DataFrame(reconstruction).groupby(['split','label']).size().rename('pairs').reset_index())

    records=[]; representative={}; primary_bytes=[]
    shards=sorted((ROOT/'data/mirror/humanitarian').glob('*.parquet'))
    for path in shards:
        split=path.name.split('-')[0].replace('validation','dev')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for item in batch.to_pylist():
                rowid=f"{split}:{item['image_id']}"
                matches=indexes[(item['event_name'],item['image_id'])]
                source=matches[0] if len(matches)==1 else None
                blob=item['image']['bytes'] if item.get('image') else None
                text=item.get('text') or ''
                lab=int(item['label']);label=LABELS[lab] if 0<=lab<len(LABELS) else 'INVALID'
                rec={'row_id':rowid,'split':split,'event_name':item['event_name'], 'tweet_id':str(item['tweet_id']),
                     'image_id':str(item['image_id']),'label_id':lab,'label':label,'tweet_text':text,
                     'source_image_path':source['image_path'] if source else '',
                     'source_text_label':source['text_human'] if source else '',
                     'source_image_label':source['image_human'] if source else '',
                     'source_text_conf':source['text_human_conf'] if source else '',
                     'source_image_conf':source['image_human_conf'] if source else '',
                     'primary_id_join_count':len(matches),'empty_text':not text.strip(),
                     'chars':len(text),'words':len(re.findall(r'\S+',text)),
                     'words_without_urls':len(re.findall(r'\S+',re.sub(r'https?://\S+',' ',text))),
                     'url_count':len(re.findall(r'https?://\S+',text)),
                     'raw_text_sha256':sha(text.encode()),'normalized_text':norm(text),
                     'mojibake_candidate':bool(re.search('[\u00c2\u00c3\u00e2]',text)),
                     'replacement_characters':text.count('\ufffd'),
                     'tweet_image_id_mismatch':str(item['tweet_id'])!=item['image_id'].rsplit('_',1)[0],
                     'embedded_filename_mismatch':pathlib.Path(item['image'].get('path') or '').stem!=item['image_id'],
                     'image_bytes':len(blob) if blob else 0,'image_sha256':sha(blob) if blob else '',
                     'decode_ok':False,'decode_error':'','width':0,'height':0,'image_format':'','pixel_sha256':'','phash64':''}
                if source:
                    rec['text_differs_from_primary']=text!=source['tweet_text']
                    rec['source_labels_agree']=source['text_human']==source['image_human']
                    rec['target_matches_mapped_text']=label==MAP.get(source['text_human'],source['text_human'])
                    rec['target_matches_mapped_image']=label==MAP.get(source['image_human'],source['image_human'])
                    rec['source_path_id_mismatch']=pathlib.Path(source['image_path']).stem!=item['image_id']
                else:
                    for k in ['text_differs_from_primary','source_labels_agree','target_matches_mapped_text','target_matches_mapped_image','source_path_id_mismatch']:rec[k]=None
                try:
                    if not blob:raise ValueError('missing image bytes')
                    with Image.open(io.BytesIO(blob)) as check:check.verify()
                    with Image.open(io.BytesIO(blob)) as im:
                        im.load(); rec['width'],rec['height']=im.size;rec['image_format']=im.format
                        rgb=im.convert('RGB');rec['pixel_sha256']=sha(f'{rgb.size}'.encode()+rgb.tobytes())
                        rec['phash64']=photohash(rgb);rec['decode_ok']=True
                    key=(label,split);rank=sha(item['image_id'].encode())
                    if key not in representative or rank<representative[key][0]:representative[key]=(rank,blob,rec.copy())
                except Exception as exc:rec['decode_error']=f'{type(exc).__name__}: {exc}'
                rec['structurally_valid']=bool(rec['decode_ok'] and not rec['empty_text'] and label!='INVALID'
                    and not rec['tweet_image_id_mismatch'] and not rec['embedded_filename_mismatch']
                    and not rec['source_path_id_mismatch'] and len(matches)==1)
                local=ROOT/'data/images'/rec['source_image_path']
                if rec['source_image_path'] and local.is_file():
                    ph=sha(local.read_bytes());primary_bytes.append({'row_id':rowid,'image_id':item['image_id'],
                        'qcri_sha256':ph,'mirror_sha256':rec['image_sha256'],'identical':ph==rec['image_sha256']})
                records.append(rec)
        print('decoded',path.name,'rows_so_far',len(records),flush=True)
    df=pd.DataFrame(records)
    assert len(df)==8079,'Mirror changed from documented row count; inspect provenance'
    df['normalized_text_sha256']=df.normalized_text.map(lambda s:sha(s.encode()) if s else '')
    df.drop(columns='normalized_text').to_csv(ROOT/'tables/benchmark_manifest.csv',index=False)
    save('primary_mirror_byte_comparison.csv',primary_bytes)
    save('source_consistency_issues.csv',df[(df.primary_id_join_count!=1)|(df.source_labels_agree!=True)|
        (df.target_matches_mapped_text!=True)|(df.target_matches_mapped_image!=True)|(df.text_differs_from_primary==True)])
    save('structural_issues.csv',df[~df.structurally_valid])
    published=[]
    for split in ['train','dev','test']:
        for i,label in enumerate(LABELS):
            s=df[(df.split==split)&(df.label==label)]
            published.append({'split':split,'label':label,'published_pairs':PUBLISHED[split][i],
                'measured_pairs':len(s),'structurally_valid':int(s.structurally_valid.sum()),
                'unique_tweets':s.tweet_id.nunique(),'delta':len(s)-PUBLISHED[split][i]})
    counts=save('published_vs_measured.csv',published)
    save('event_distribution.csv',df.groupby(['event_name','split']).size().unstack(fill_value=0).reset_index())
    save('text_length_summary.csv',df.groupby('split')[['chars','words','words_without_urls']].describe(percentiles=[.05,.25,.5,.75,.95,.99]).stack(0,future_stack=True).reset_index())
    save('image_size_summary.csv',df.groupby('split')[['width','height','image_bytes']].describe().stack(0,future_stack=True).reset_index())
    recon=pd.DataFrame(reconstruction)
    selected=set(zip(df.split,df.image_id))
    save('reconstruction_extra_rows.csv',[x for x in reconstruction if (x['split'],x['image_id']) not in selected])
    reconids=set(zip(recon.split,recon.image_id))
    save('mirror_absent_from_reconstruction.csv',df[[x not in reconids for x in zip(df.split,df.image_id)]])

    duplicates=[]; cross=[]; duplicate_summary=[]
    for field in ['tweet_id','image_id','image_sha256','pixel_sha256','raw_text_sha256','normalized_text_sha256']:
        groups=[g for val,g in df.groupby(field,sort=True) if val and len(g)>1]
        xgroups=[g for g in groups if g.split.nunique()>1]
        duplicate_summary.append({'criterion':field,'duplicate_groups':len(groups),'duplicate_rows':sum(len(g) for g in groups),
             'excess_rows':sum(len(g)-1 for g in groups),'cross_split_groups':len(xgroups),
             'cross_split_rows':sum(len(g) for g in xgroups)})
        for n,g in enumerate(groups):
            ids=g.row_id.tolist();splits='|'.join(sorted(g.split.unique())); labs='|'.join(sorted(g.label.unique()))
            for _,r in g.iterrows():duplicates.append({'criterion':field,'group_id':f'{field}:{n}',
                'row_id':r.row_id,'split':r.split,'image_id':r.image_id,'group_size':len(g),
                'group_splits':splits,'cross_split':g.split.nunique()>1,'group_labels':labs})
            if g.split.nunique()>1:
                for a,b in itertools.combinations(g.to_dict('records'),2):
                    if a['split']!=b['split']:cross.append({'criterion':field,'row_a':a['row_id'],'row_b':b['row_id'],
                         'image_a':a['image_id'],'image_b':b['image_id'],'split_a':a['split'],'split_b':b['split'],
                         'same_label':a['label']==b['label']})
    save('duplicate_summary.csv',duplicate_summary)
    save('duplicate_groups.csv',duplicates)
    save('cross_split_exact_matches.csv',cross,columns=['criterion','row_a','row_b','image_a','image_b','split_a','split_b','same_label'])

    # BK tree indexes unique 64-bit perceptual hashes. These are candidates, not confirmed duplicates.
    hashes=collections.defaultdict(list)
    for i,r in df[df.decode_ok].iterrows():hashes[int(r.phash64,16)].append(i)
    tree=None;near=[]
    def query(node,value,threshold=4):
        if node is None:return []
        distance=(node[0]^value).bit_count();hits=[node[0]] if distance<=threshold else []
        for edge,child in node[1].items():
            if distance-threshold<=edge<=distance+threshold:hits.extend(query(child,value,threshold))
        return hits
    def insert(node,value):
        if node is None:return [value,{}]
        cur=node
        while True:
            distance=(cur[0]^value).bit_count()
            if distance==0:return node
            if distance not in cur[1]:cur[1][distance]=[value,{}];return node
            cur=cur[1][distance]
    for h in sorted(hashes):
        for other in query(tree,h)+[h]:
            for ia in hashes[h]:
                for ib in hashes[other]:
                    if ia==ib or (h==other and ia>ib):continue
                    a=df.loc[ia];b=df.loc[ib]
                    if a.split!=b.split and a.pixel_sha256!=b.pixel_sha256:
                        near.append({'row_a':a.row_id,'row_b':b.row_id,'image_a':a.image_id,'image_b':b.image_id,
                                     'split_a':a.split,'split_b':b.split,'phash_distance':(h^other).bit_count(),
                                     'same_label':a.label==b.label})
        tree=insert(tree,h)
    save('near_duplicate_candidates.csv',near,columns=['row_a','row_b','image_a','image_b','split_a','split_b','phash_distance','same_label'])

    examples=[]
    for (label,split),(rank,blob,rec) in representative.items():
        if split=='dev':continue
        dest=ROOT/'examples'/f'{rec["label_id"]}_{split}_{rec["image_id"]}.jpg';dest.write_bytes(blob)
        examples.append({**{k:rec[k] for k in ['row_id','split','label','label_id','event_name','tweet_id','image_id','tweet_text','source_text_label','source_image_label']},
                         'file':str(dest.relative_to(ROOT)),'selection':'minimum SHA256(image_id) within class and split; deterministic, not hand-picked'})
    save('representative_examples.csv',examples)
    ntest=df[df.split=='test'].groupby('label').size()
    save('minority_reliability.csv',[{'label':lab,'test_pairs':int(n),'one_error_changes_recall_percentage_points':100/n,
          'best_case_wilson_lower_95_if_all_correct':n/(n+1.959963984540054**2),
          'note':'Hypothetical sampling resolution, not model performance'} for lab,n in ntest.items()])
    summary={
      'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'qcri_revision':'10a5626ba112ab9c50369c473a2062bcf205d708',
      'mirror_revision':'a935fda3cee8a5250746a8d1d6278214d615cc89',
      'primary_rows':len(raw),'mirror_pairs':len(df),'primary_reconstructed_pairs':len(reconstruction),
      'official_agreed_zip_access':'HTTP 403; no original agreed ZIP bytes inspected',
      'decoded_images':int(df.decode_ok.sum()),'structurally_valid':int(df.structurally_valid.sum()),
      'empty_texts':int(df.empty_text.sum()),'id_mismatches':int(df.tweet_image_id_mismatch.sum()),
      'filename_mismatches':int(df.embedded_filename_mismatch.sum()),'primary_join_issues':int((df.primary_id_join_count!=1).sum()),
      'source_label_disagreements':int((df.source_labels_agree==False).sum()),
      'target_differs_from_primary_text':int((df.target_matches_mapped_text==False).sum()),
      'target_differs_from_primary_image':int((df.target_matches_mapped_image==False).sum()),
      'texts_differ_from_primary':int((df.text_differs_from_primary==True).sum()),
      'mojibake_candidates':int(df.mojibake_candidate.sum()),'replacement_character_rows':int((df.replacement_characters>0).sum()),
      'unique_tweets':int(df.tweet_id.nunique()),'unique_image_bytes':int(df.image_sha256.nunique()),
      'image_bytes_total':int(df.image_bytes.sum()),'primary_image_comparisons':len(primary_bytes),
      'primary_image_byte_matches':sum(x['identical'] for x in primary_bytes),
      'all_published_cells_match':bool((counts.delta==0).all()),
      'duplicates':duplicate_summary,'phash_cross_split_candidate_pairs':len(near),
      'text_words':{split:{'median':float(g.words.median()),'p95':float(g.words.quantile(.95)),
                         'min':int(g.words.min()),'max':int(g.words.max())} for split,g in df.groupby('split')},
      'duration_seconds':round(time.monotonic()-started,2),
      'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'pillow':Image.__version__},
      'no_training_performed':True,
    }
    (ROOT/'provenance/audit_summary.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'provenance/label_mapping.json').write_text(json.dumps({'labels_in_id_order':LABELS,'merge_after_agreement':MAP,
      'agreement_rule':'source text_human == image_human BEFORE merging; authors agreed archive remains unverified',
      'do_not_substitute':'Do not filter on QCRI humanitarian JSON label alone, or agree only after merging.'},indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
