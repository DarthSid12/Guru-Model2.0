import csv,json,statistics,math
from pathlib import Path
from collections import Counter,defaultdict
from execution.common.io import seeds,rate,safe_name,new_output,write_json,sha
from execution.common.features import load_probabilities
from .matching import evaluate

def validate(c):
    seeds(c['seeds']);safe_name(c['category'])
    if min(c['num_identities'],c['images_per_identity'])<2 or min(c['num_fixations'],c['threads'])<1:raise ValueError('Need >=2 identities and photos/identity, positive fixations/threads')
    if not 0<=c['calibration_target']<=1:raise ValueError('Calibration target must be a fraction')
    if not c['models'] or len({m['name'] for m in c['models']})!=len(c['models']):raise ValueError('Models need unique names')
    if len({m['cache_subdir'] for m in c['models']})!=len(c['models']):raise ValueError('Each model needs a distinct cache_subdir')
    for m in c['models']:
        safe_name(m['name']);safe_name(m['cache_subdir'])
        if m['noise'] is not None:rate(m['noise'])

def run(c):
    import torch
    validate(c);torch.set_num_threads(c['threads']);out=new_output(c['output'],c);rows=[];provenance={}
    for m in c['models']:
        probs,receipt=load_probabilities(m,{'faces':c['category']},c,c['num_fixations']);provenance[m['name']]=receipt
        meta=Path(c['cache_root'])/m['cache_subdir']/'METADATA.json'
        if meta.exists():labels=json.loads(meta.read_text())['faces']['labels']
        else:labels=json.loads((Path(c['packed_root'])/c['category']/'valid/meta.json').read_text())['labels']
        counts=Counter(labels)
        if len(counts)!=c['num_identities'] or set(counts.values())!={c['images_per_identity']}:raise ValueError('The supplied bank must exactly match the requested identity/photo counts; no silent resampling.')
        n=len(labels);expected=n*(c['images_per_identity']-1)*(n-c['images_per_identity'])
        if any(len(a)!=n for a in probs.values()):raise ValueError('Probability and identity-label rows differ')
        p=m['noise']
        if p is None:
            curve={}
            for candidate in [i/20 for i in range(11)]:
                correct,trials=evaluate(probs['faces','U'][:,:c['num_fixations']],labels,c['calibration_seed'],candidate);curve[candidate]=correct/trials
            best=min(curve,key=lambda v:(abs(curve[v]-c['calibration_target']),v))
            for candidate in sorted({round(best+i*.01,2) for i in range(-4,5) if 0<=round(best+i*.01,2)<=.5}):
                if candidate not in curve:
                    correct,trials=evaluate(probs['faces','U'][:,:c['num_fixations']],labels,c['calibration_seed'],candidate);curve[candidate]=correct/trials
            p=min(curve,key=lambda v:(abs(curve[v]-c['calibration_target']),v))
            write_json(out/f'{m["name"]}-calibration.json',dict(noise=p,target=c['calibration_target'],achieved=curve[p],seed=c['calibration_seed'],curve=curve,target_within_sampled_range=min(curve.values())<=c['calibration_target']<=max(curve.values())))
        for seed in seeds(c['seeds']):
            for condition in ['U','I']:
                correct,n=evaluate(probs['faces',condition][:,:c['num_fixations']],labels,seed,p)
                if n!=expected:raise ValueError('Incorrect triplet count')
                rows.append(dict(model=m['name'],seed=seed,condition=condition,noise=p,identities=c['num_identities'],images_per_identity=c['images_per_identity'],fixations=c['num_fixations'],trials=n,correct=correct,accuracy=100*correct/n))
        print(m['name'],'Kanwisher complete',flush=True)
    with (out/'all_records.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r['model'],r['condition']].append(r['accuracy'])
    summary=[]
    for key,vals in groups.items():
        sd=statistics.stdev(vals) if len(vals)>1 else None
        summary.append(dict(model=key[0],condition=key[1],mean=statistics.mean(vals),sd=sd,sem=sd/math.sqrt(len(vals)) if sd is not None else None))
    write_json(out/'SUMMARY.json',dict(units='accuracy percent',uncertainty='sample SD/SEM across simulation seeds',cells=summary));write_json(out/'PROVENANCE.json',provenance);write_json(out/'COMPLETE.json',dict(rows=len(rows),sha256=sha(out/'all_records.csv')))
