import csv,json,math,os,statistics as st,subprocess,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from common import ROOT,sha,atomic,DESIGNS

def verify():
    s=json.loads((ROOT/'SPEC.json').read_text())
    for n,h in s['files'].items():assert sha(ROOT/n)==h,n
    for n,h in s['external_files'].items():assert sha(n)==h,n
    return s

def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path.with_suffix('.tmp').replace(path)

def fit_axis(e,s,sigma,condition,target,test_held=None):
    from kde import inputs,score
    results={}
    def accuracy(rate):return sum(x['correct'] for x in results[rate])/(50*12)
    def best(rates):return min(rates,key=lambda p:(abs(accuracy(p)-target),p))
    def evaluate(rates):
        missing=[r for r in rates if r not in results]
        for r in missing:results[r]=[]
        for seed in s['calibration_seeds']:
            e.draws.clear()
            for rate in missing:
                tn=rate if test_held is None else test_held
                p,mem,q=inputs(e,seed,condition,rate,tn)
                indices=[j for i,t in enumerate(p['trials']) if t['category']=='faces' for j in [2*i,2*i+1]]
                values,_=score(q[indices],mem,sigma);ok=values.reshape(12,2)[:,0]>values.reshape(12,2)[:,1]
                results[rate].append(dict(seed=seed,n_trials=12,correct=int(ok.sum())))
    evaluate(s['coarse_noise_grid']);coarse=best(s['coarse_noise_grid'])
    fine=[i/1000 for i in range(max(0,round(coarse*1000)-50),min(500,round(coarse*1000)+50)+1,5)]
    evaluate(fine);fine_best=best(results)
    ultra=[i/1000 for i in range(max(0,round(fine_best*1000)-5),min(500,round(fine_best*1000)+5)+1)]
    evaluate(ultra);selected=best(results)
    curve=[dict(rate=r,accuracy=accuracy(r),seed_counts=results[r]) for r in sorted(results)]
    return dict(condition=condition,target=target,test_noise_held_at=test_held,coarse_best=coarse,fine_grid=fine,fine_best=fine_best,ultrafine_grid=ultra,selected_noise=selected,achieved=accuracy(selected),absolute_target_error=abs(accuracy(selected)-target),target_within_sampled_range=min(x['accuracy'] for x in curve)<=target<=max(x['accuracy'] for x in curve),selected_boundary=selected in [0.,.5],curve=curve)

def worker(index):
    import torch
    from engine import Experiment
    from kde import inputs,score,noise_assignment
    from design import plan
    torch.set_num_threads(1);s=verify();m=s['models'][index];v=m['record']['variant'];old1,old2=m['original_two_noise']['p1'],m['original_two_noise']['p2']
    baseline={}
    for r in csv.DictReader((Path(s['parent_root'])/'results/all_trials.csv').open()):
        if r['model']==m['record']['model'] and r['design']=='faces_houses' and r['protocol']=='study_only_fitted' and r['condition'] in ['UU','II']:
            baseline[int(r['simulation_seed']),r['condition'],r['category']]=int(r['correct_pairs'])
    assert len(baseline)==200
    e=Experiment(m['cache'],'faces_houses');replayed=0
    for seed in s['calibration_seeds']:
        e.draws.clear()
        for cond in ['UU','II']:
            sn,tn=noise_assignment(cond,old1,old2);p,mem,q=inputs(e,seed,cond,sn,tn);values,_=score(q,mem,2.)
            ok=values.reshape(24,2)[:,0]>values.reshape(24,2)[:,1]
            for cat in ['faces','houses']:
                count=sum(int(ok[i]) for i,t in enumerate(p['trials']) if t['category']==cat)
                assert count==baseline[seed,cond,cat];replayed+=1
    p1fit=fit_axis(e,s,m['new_sigma'],'UU',s['targets']['UU'])
    print('FITTED UU',v,p1fit['selected_noise'],p1fit['achieved'],flush=True)
    p1=p1fit['selected_noise'];p2fit=fit_axis(e,s,m['new_sigma'],'II',s['targets']['II'],p1);p2=p2fit['selected_noise']
    print('FITTED II',v,p2,p2fit['achieved'],flush=True)
    fit=dict(spec_sha256=sha(ROOT/'SPEC.json'),model=m['record'],sigma=m['new_sigma'],calibration_design='faces_houses',calibration_seeds=s['calibration_seeds'],p1=p1,p2=p2,UU_fit=p1fit,II_fit=p2fit,noise_rule='UU/UI(p1,p1),IU/II(p2,p1),study/test',not_fitted=['UI','IU','houses','objects','validation'])
    atomic(ROOT/'calibration'/(v+'.json'),fit)
    settings=[('new_sigma_two_noise',m['new_sigma'],p1,p2),('sigma2_two_noise_originalfit',2.,old1,old2),('sigma2_two_noise_matchedfit',2.,p1,p2),('new_sigma_single_p1',m['new_sigma'],p1,p1)]
    saved=json.loads((ROOT/'STIMULUS_PLANS.json').read_text());rows=[];trials=[]
    for design in s['designs']:
        e=Experiment(m['cache'],design)
        for seed in s['validation_seeds']:
            e.draws.clear();p=plan(design,seed);assert json.loads(json.dumps(p))==saved[design][str(seed)]
            for cond in s['conditions']:
                encoded={};scored={}
                for method,sigma,a,b in settings:
                    sn,tn=noise_assignment(cond,a,b);key=(sn,tn)
                    if key not in encoded:encoded[key]=inputs(e,seed,cond,sn,tn)[1:]
                    if (sn,tn,sigma) not in scored:
                        mem,q=encoded[key];scored[sn,tn,sigma]=score(q,mem,sigma)
                    values,win=scored[sn,tn,sigma];ok=values.reshape(24,2)[:,0]>values.reshape(24,2)[:,1]
                    for cat in DESIGNS[design]:
                        indices=[i for i,t in enumerate(p['trials']) if t['category']==cat]
                        mask=''.join('1' if ok[i] else '0' for i in indices);correct=mask.count('1')
                        nself=0;ncross=0
                        for ti in indices:
                            t=p['trials'][ti];oi=2*ti;ni=oi+1;wc,wr=p['study_order'][int(win[oi])];vc,vr=p['study_order'][int(win[ni])]
                            identity=(wc,wr)==(cat,t['old']);nself+=int(identity);ncross+=int(wc!=cat)
                            trials.append(dict(variant=v,design=design,method=method,sigma=sigma,p1=a,p2=b,simulation_seed=seed,condition=cond,category=cat,study_noise=sn,test_noise=tn,old_row=t['old'],new_row=t['new'],old_log_familiarity=float(values[oi]),novel_log_familiarity=float(values[ni]),correct=int(ok[ti]),old_winner_category=wc,old_winner_row=wr,novel_winner_category=vc,novel_winner_row=vr,old_identity_correct=int(identity)))
                        rows.append(dict(model=m['record']['model'],variant=v,checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,design=design,method=method,sigma=sigma,p1=a,p2=b,study_noise=sn,test_noise=tn,simulation_seed=seed,condition=cond,category=cat,n_trials=12,correct_pairs=correct,errors=12-correct,accuracy_pct=100*correct/12,correct_mask=mask,old_correct_memory_count=nself,old_cross_category_memory_count=ncross))
            if seed%10==0:print('PROGRESS',v,design,seed-200,'/50',flush=True)
    assert len(rows)==3200 and len(trials)==38400 and replayed==200
    csvwrite(ROOT/'results'/(v+'-raw.csv'),rows);csvwrite(ROOT/'results'/(v+'-trials.csv'),trials)
    atomic(ROOT/'results'/(v+'-DONE.json'),dict(spec_sha256=sha(ROOT/'SPEC.json'),rows=len(rows),trial_rows=len(trials),original_control_records_replayed=replayed,csv_sha256=sha(ROOT/'results'/(v+'-raw.csv')),trials_sha256=sha(ROOT/'results'/(v+'-trials.csv')),calibration_sha256=sha(ROOT/'calibration'/(v+'.json'))))

def moments(a):
    sd=st.stdev(a);return dict(n=len(a),mean=st.mean(a),sd=sd,sem=sd/math.sqrt(len(a)))

def aggregate():
    s=verify();rows=[];trials=[];groups=defaultdict(list);paired=defaultdict(dict)
    for m in s['models']:
        v=m['record']['variant'];done=json.loads((ROOT/'results'/(v+'-DONE.json')).read_text())
        assert done['spec_sha256']==sha(ROOT/'SPEC.json') and done['original_control_records_replayed']==200
        assert done['csv_sha256']==sha(ROOT/'results'/(v+'-raw.csv')) and done['trials_sha256']==sha(ROOT/'results'/(v+'-trials.csv')) and done['calibration_sha256']==sha(ROOT/'calibration'/(v+'.json'))
        data=list(csv.DictReader((ROOT/'results'/(v+'-raw.csv')).open()));rows.extend(data);trials.extend(list(csv.DictReader((ROOT/'results'/(v+'-trials.csv')).open())))
        for r in data:
            assert int(r['errors'])+int(r['correct_pairs'])==12 and r['correct_mask'].count('1')==int(r['correct_pairs'])
            key=(v,r['design'],r['method'],r['category'],r['condition']);groups[key].append(r)
            pair=(v,r['design'],r['category'],r['condition'],int(r['simulation_seed']));assert r['method'] not in paired[pair];paired[pair][r['method']]=int(r['errors'])
    assert len(rows)==6400 and len(trials)==76800 and len(groups)==128
    cells=[]
    for k,values in groups.items():
        assert sorted(int(r['simulation_seed']) for r in values)==s['validation_seeds']
        cells.append(dict(zip(['variant','design','method','category','condition'],k),sigma=float(values[0]['sigma']),p1=float(values[0]['p1']),p2=float(values[0]['p2']),study_noise=float(values[0]['study_noise']),test_noise=float(values[0]['test_noise']),errors=moments([int(r['errors']) for r in values]),old_identity_accuracy_pct=moments([100*int(r['old_correct_memory_count'])/12 for r in values]),old_cross_category_match_pct=moments([100*int(r['old_cross_category_memory_count'])/12 for r in values])))
    effects=defaultdict(list)
    for k,values in paired.items():
        assert set(values)==set(s['methods'])
        for method in s['methods'][1:]:effects[k[:-1]+(method,)].append(values[method]-values['new_sigma_two_noise'])
    comparison=[dict(zip(['variant','design','category','condition','control'],k),paired_improvement_errors=moments(a)) for k,a in effects.items()]
    csvwrite(ROOT/'results/all_records.csv',rows);csvwrite(ROOT/'results/all_trials.csv',trials)
    atomic(ROOT/'results/SUMMARY.json',dict(cells=cells,paired_comparisons=comparison,uncertainty='Mean/sampleSD/SEM across50new simulation seeds201-250;fixed trainingseed46;allcategories12pairs'))
    atomic(ROOT/'results/VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),rows=6400,trial_rows=76800,cells=128,seeds_per_cell=50,parent_control_records_replayed=400,csv_sha256=sha(ROOT/'results/all_records.csv'),trials_sha256=sha(ROOT/'results/all_trials.csv'),summary_sha256=sha(ROOT/'results/SUMMARY.json')))

def main():
    s=verify();r=json.loads((ROOT/'RUN.json').read_text());assert r['registration_complete'] and r['spec_sha256']==sha(ROOT/'SPEC.json') and r['execution_id']==os.environ['GURU_RUN_ID']
    from tests import run
    run(s)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'worker',str(i)],check=True),range(2)))
    aggregate();print('NEW-SIGMA TWO-NOISE YIN COMPLETE;400ORIGINAL CONTROLS EXACT',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:main()
