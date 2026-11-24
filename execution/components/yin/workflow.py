import csv,json,math,statistics
from pathlib import Path
from collections import defaultdict
from execution.common.io import seeds,rate,new_output,sha,write_json,safe_name
from execution.common.features import load_probabilities
from .design import make_plan
from .memory import Memory,noise_rates,familiarity
from .calibration import fit_axis

def validate(c):
    seeds(c['seeds']);seeds(c['calibration_seeds'])
    if c['sigma']<=0 or min(c['study_fixations'],c['test_fixations'],c['threads'])<1:raise ValueError('sigma, fixation counts and threads must be positive')
    if c['noise_protocol'] not in ['single_uu','study_only_fitted','two_cross','two_shared_ui']:raise ValueError('Unknown noise protocol')
    if c['readout'] not in ['identity_max','category_max','pooled_category','pooled_all']:raise ValueError('Unknown readout')
    if not c['conditions'] or not set(c['conditions'])<=set(['UU','UI','IU','II']):raise ValueError('Invalid conditions')
    if not c['models'] or len({m['name'] for m in c['models']})!=len(c['models']):raise ValueError('Models must have unique names')
    if len({m['cache_subdir'] for m in c['models']})!=len(c['models']):raise ValueError('Each model needs a distinct cache_subdir')
    for m in c['models']:
        safe_name(m['name']);safe_name(m['cache_subdir'])
        if not c['calibrate']:
            rate(m['noise'])
            if c['noise_protocol']!='single_uu':rate(m['noise2'])
    for name,definition in c['designs'].items():
        safe_name(name)
        if not definition:raise ValueError('Empty study design')
        for cat,n in definition.items():
            if cat not in c['banks'] or set(n)!=set(['pool','study','trials']):raise ValueError('Each category needs pool, study and trials plus a bank mapping')
            if any(not isinstance(v,int) or v<1 for v in n.values()) or n['pool']!=n['study']+n['trials'] or n['trials']>n['study']:raise ValueError('Require pool=study+trials, and 1 <= trials <= study')
    if c['calibrate'] and (c['calibration_design'] not in c['designs'] or c['calibration_category'] not in c['designs'][c['calibration_design']]):raise ValueError('Calibration anchor is not present in the calibration design')
    for v in c['targets'].values():
        if not 0<=v<=1:raise ValueError('Targets are fractions in [0,1]')

def run(c):
    import torch
    validate(c);torch.set_num_threads(c['threads']);out=new_output(c['output'],c);rows=[];plans={};provenance={}
    for m in c['models']:
        if c['verify_registry']:
            registry=json.loads((Path(__file__).resolve().parents[3]/'yin_tests/current_kde/registry.json').read_text());name=m['cache_subdir']
            if name not in registry['cache_sha256']:raise ValueError('Unknown cache in pinned registry. Set verify_registry=false explicitly for a new model.')
            for file,digest in registry['cache_sha256'][name].items():
                if sha(Path(c['cache_root'])/name/file)!=digest:raise ValueError(f'Pinned cache mismatch: {file}')
        needed={cat:bank for cat,bank in c['banks'].items() if any(cat in d for d in c['designs'].values())}
        probs,receipt=load_probabilities(m,needed,c,max(c['study_fixations'],c['test_fixations']));provenance[m['name']]=receipt
        sizes={cat:len(probs[cat,'U']) for cat in needed}
        for d in c['designs'].values():
            for cat,n in d.items():
                if n['pool']>sizes[cat]:raise ValueError(f'{cat}: requested pool exceeds bank size')
        def measure(memory,plan,condition,sn,tn):
            mem,q=memory.inputs(plan,condition,sn,tn,c['study_fixations'],c['test_fixations'])
            scores=familiarity(q,mem,c['sigma'],c['readout'],plan).reshape(-1,2)
            return (scores[:,0]>scores[:,1]).tolist()
        p1=m['noise'];p2=m.get('noise2')
        if c['calibrate']:
            design=c['calibration_design'];anchor=c['calibration_category'];memory=Memory(probs,design)
            def evaluate(condition,sn,tn):
                correct=total=0
                for seed in seeds(c['calibration_seeds']):
                    memory.draws.clear();p=make_plan(design,c['designs'][design],seed,sizes);ok=measure(memory,p,condition,sn,tn)
                    for i,t in enumerate(p['trials']):
                        if t['category']==anchor:correct+=ok[i];total+=1
                return correct/total
            first=fit_axis(evaluate,'UU',c['targets']['UU'],None,c['noise_protocol']);p1=first['noise'];fits={'UU':first}
            if c['noise_protocol']!='single_uu':
                condition='II'
                second=fit_axis(evaluate,condition,c['targets'][condition],p1,c['noise_protocol']);p2=second['noise'];fits[condition]=second
            write_json(out/f'{m["name"]}-calibration.json',fits)
        for name,definition in c['designs'].items():
            memory=Memory(probs,name)
            for seed in seeds(c['seeds']):
                memory.draws.clear();plan=make_plan(name,definition,seed,sizes);plans[f'{name}:{seed}']=plan
                for cond in c['conditions']:
                    sn,tn=noise_rates(c['noise_protocol'],cond,p1,p2);ok=measure(memory,plan,cond,sn,tn)
                    for cat in definition:
                        mask=''.join(str(int(ok[i])) for i,t in enumerate(plan['trials']) if t['category']==cat)
                        rows.append(dict(model=m['name'],design=name,category=cat,seed=seed,condition=cond,readout=c['readout'],sigma=c['sigma'],noise_protocol=c['noise_protocol'],noise=p1,noise2=p2,study_noise=sn,test_noise=tn,trials=len(mask),correct=mask.count('1'),correct_mask=mask,accuracy=100*mask.count('1')/len(mask)))
            print(m['name'],name,'complete',flush=True)
    with (out/'all_records.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r['model'],r['design'],r['category'],r['condition']].append(r['accuracy'])
    summary=[]
    for key,values in groups.items():
        sd=statistics.stdev(values) if len(values)>1 else None
        summary.append(dict(zip(['model','design','category','condition'],key),n=len(values),mean=statistics.mean(values),sd=sd,sem=sd/math.sqrt(len(values)) if sd is not None else None))
    write_json(out/'SUMMARY.json',dict(units='accuracy percent',uncertainty='sample SD/SEM across simulation seeds',calibration_evaluation_seed_overlap=sorted(set(seeds(c['seeds']))&set(seeds(c['calibration_seeds']))),cells=summary))
    write_json(out/'PROVENANCE.json',provenance);write_json(out/'STIMULUS_PLANS.json',plans);write_json(out/'COMPLETE.json',dict(rows=len(rows),sha256=sha(out/'all_records.csv')))
