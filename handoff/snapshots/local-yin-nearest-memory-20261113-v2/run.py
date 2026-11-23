import csv,json,math,os,statistics as st,subprocess,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import ROOT,sha,atomic,DESIGNS

def verify():
    s=json.loads((ROOT/'SPEC.json').read_text())
    for n,h in s['files'].items():assert sha(ROOT/n)==h,n
    for n,h in s['external_files'].items():assert sha(n)==h,n
    return s

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path.with_suffix('.tmp').replace(path)

def calibrate(e,s):
    from retrieval import nearest_memory,encode_inputs
    results={};target=s['face_UU_target_accuracy'];n=DESIGNS[e.design]['faces']['trials']
    def evaluate(rates):
        missing=[p for p in rates if p not in results]
        for p in missing:results[p]=[]
        for seed in s['calibration_seeds']:
            e.draws.clear()
            for noise in missing:
                p,mem,q=encode_inputs(e,seed,'UU',noise);score,_,_=nearest_memory(q,mem)
                correct=score.reshape(24,2)[:,0]>score.reshape(24,2)[:,1]
                value=sum(int(correct[i]) for i,t in enumerate(p['trials']) if t['category']=='faces')
                results[noise].append(dict(seed=seed,correct=value,n_trials=n))
    def achieved(rate):return sum(r['correct'] for r in results[rate])/(len(s['calibration_seeds'])*n)
    evaluate(s['calibration_coarse_rates'])
    coarse_best=min(s['calibration_coarse_rates'],key=lambda p:(abs(achieved(p)-target),p))
    fine=[i/1000 for i in range(max(0,round(coarse_best*1000)-50),min(500,round(coarse_best*1000)+50)+1,5)]
    evaluate(fine)
    best=min(results,key=lambda p:(abs(achieved(p)-target),p))
    curve=[dict(noise_rate=p,accuracy=achieved(p),seed_counts=results[p]) for p in sorted(results)]
    low=min(c['accuracy'] for c in curve);high=max(c['accuracy'] for c in curve)
    return dict(spec_sha256=sha(ROOT/'SPEC.json'),design=e.design,target_face_UU_accuracy=target,coarse_best=coarse_best,fine_rates=fine,selected_noise=best,achieved_face_UU_accuracy=achieved(best),absolute_target_error=abs(achieved(best)-target),target_within_sampled_range=low<=target<=high,selected_boundary=best in [0.,.5],calibration_seeds=s['calibration_seeds'],curve=curve,held_out_outcomes=['house','object','faceII','faceUI','faceIU'])

def worker(index):
    import torch
    from engine import Experiment
    from retrieval import nearest_memory,kde_legacy,encode_inputs
    from design import plan
    torch.set_num_threads(1);s=verify();m=s['models'][index];variant=m['record']['variant']
    parent={}
    for r in csv.DictReader((Path(s['parent_root'])/'results/all_trials.csv').open()):
        if r['model']==m['record']['model'] and r['design'] in s['designs'] and r['protocol']=='single_uu' and r['condition']=='UU':
            parent[r['design'],int(r['simulation_seed']),r['category']]=int(r['correct_pairs'])
    assert len(parent)==250
    saved=json.loads((ROOT/'STIMULUS_PLANS.json').read_text());rows=[];trials=[];replayed=0
    for design in s['designs']:
        e=Experiment(m['cache'],design);oldrate=m['baseline_noise'][design]
        for seed in s['calibration_seeds']:
            e.draws.clear();p,mem,q=encode_inputs(e,seed,'UU',oldrate);score,_,_=kde_legacy(q,mem)
            ok=score.reshape(24,2)[:,0]>score.reshape(24,2)[:,1]
            for cat,n in DESIGNS[design].items():
                count=sum(int(ok[i]) for i,t in enumerate(p['trials']) if t['category']==cat)
                assert count==parent[design,seed,cat],(variant,design,seed,cat);replayed+=1
        calibration=calibrate(e,s);newrate=calibration['selected_noise']
        atomic(ROOT/'calibration'/(variant+'-'+design+'.json'),calibration)
        print('CALIBRATED',variant,design,newrate,calibration['achieved_face_UU_accuracy'],flush=True)
        methods=[('nearest_memory',newrate),('kde_original_face_fit',oldrate),('kde_at_nearest_noise',newrate)]
        for seed in s['validation_seeds']:
            e.draws.clear();p=plan(design,seed)
            assert json.loads(json.dumps(p))==saved[design][str(seed)]
            for condition in s['conditions']:
                encoded={};kde={}
                for noise in set([oldrate,newrate]):encoded[noise]=encode_inputs(e,seed,condition,noise)[1:]
                for method,noise in methods:
                    mem,q=encoded[noise]
                    if method=='nearest_memory':
                        score,winners,d=nearest_memory(q,mem);per_memory=-d
                    else:
                        if noise not in kde:kde[noise]=kde_legacy(q,mem)
                        score,winners,per_memory=kde[noise]
                    correct=score.reshape(24,2)[:,0]>score.reshape(24,2)[:,1]
                    for cat,n in DESIGNS[design].items():
                        indices=[i for i,t in enumerate(p['trials']) if t['category']==cat]
                        mask=''.join('1' if correct[i] else '0' for i in indices);num=mask.count('1');count=n['trials']
                        n_id=0;n_wrong_category=0
                        for ti in indices:
                            t=p['trials'][ti];oi=2*ti;ni=oi+1;win=int(winners[oi]);wc,wr=p['study_order'][win];truth=p['study_order'].index((cat,t['old']))
                            idok=(wc,wr)==(cat,t['old']);n_id+=int(idok);n_wrong_category+=int(wc!=cat)
                            nc,nr=p['study_order'][int(winners[ni])]
                            trials.append(dict(variant=variant,design=design,method=method,noise_rate=noise,simulation_seed=seed,condition=condition,category=cat,old_row=t['old'],new_row=t['new'],correct=int(correct[ti]),old_score=float(score[oi]),new_score=float(score[ni]),old_winning_category=wc,old_winning_row=wr,new_winning_category=nc,new_winning_row=nr,old_identity_correct=int(idok),old_true_memory_score=float(per_memory[truth,oi]),old_winner_tie_count=int((per_memory[:,oi]==score[oi]).sum())))
                        rows.append(dict(model=m['record']['model'],variant=variant,checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,design=design,method=method,noise_rate=noise,simulation_seed=seed,condition=condition,category=cat,n_trials=count,correct_pairs=num,errors=count-num,error_points_12=12*(count-num)/count,accuracy_pct=100*num/count,correct_mask=mask,old_correct_memory_count=n_id,old_cross_category_memory_count=n_wrong_category))
            if seed%10==0:print('PROGRESS',variant,design,seed-150,'/50',flush=True)
    assert replayed==250 and len(rows)==3000 and len(trials)==28800
    write_csv(ROOT/'results'/(variant+'-raw.csv'),rows);write_csv(ROOT/'results'/(variant+'-trials.csv'),trials)
    atomic(ROOT/'results'/(variant+'-DONE.json'),dict(spec_sha256=sha(ROOT/'SPEC.json'),rows=len(rows),trial_rows=len(trials),original_KDE_calibration_UU_category_records_replayed=replayed,csv_sha256=sha(ROOT/'results'/(variant+'-raw.csv')),trials_sha256=sha(ROOT/'results'/(variant+'-trials.csv'))))

def moments(v):
    sd=st.stdev(v);return dict(n=len(v),mean=st.mean(v),sd=sd,sem=sd/math.sqrt(len(v)))

def aggregate():
    s=verify();rows=[];trials=[];groups=defaultdict(list);paired=defaultdict(dict)
    for m in s['models']:
        v=m['record']['variant'];done=json.loads((ROOT/'results'/(v+'-DONE.json')).read_text())
        assert done['spec_sha256']==sha(ROOT/'SPEC.json') and done['original_KDE_calibration_UU_category_records_replayed']==250
        assert done['csv_sha256']==sha(ROOT/'results'/(v+'-raw.csv')) and done['trials_sha256']==sha(ROOT/'results'/(v+'-trials.csv'))
        data=list(csv.DictReader((ROOT/'results'/(v+'-raw.csv')).open()));rows.extend(data)
        trials.extend(list(csv.DictReader((ROOT/'results'/(v+'-trials.csv')).open())))
        for r in data:
            key=(v,r['design'],r['method'],r['category'],r['condition']);seed=int(r['simulation_seed'])
            assert r['correct_mask'].count('1')==int(r['correct_pairs']) and int(r['errors'])+int(r['correct_pairs'])==int(r['n_trials'])
            groups[key].append(r);paired[(v,r['design'],r['category'],r['condition'],seed)][r['method']]=float(r['error_points_12'])
    assert len(rows)==6000 and len(trials)==57600 and len(groups)==120
    cells=[]
    for k,values in groups.items():
        assert sorted(int(r['simulation_seed']) for r in values)==s['validation_seeds']
        cells.append(dict(zip(['variant','design','method','category','condition'],k),noise_rate=float(values[0]['noise_rate']),n_trials_per_seed=int(values[0]['n_trials']),error_points_12=moments([float(r['error_points_12']) for r in values]),actual_errors=moments([int(r['errors']) for r in values]),old_identity_accuracy_pct=moments([100*int(r['old_correct_memory_count'])/int(r['n_trials']) for r in values]),old_cross_category_match_pct=moments([100*int(r['old_cross_category_memory_count'])/int(r['n_trials']) for r in values])))
    differences=defaultdict(list)
    for key,value in paired.items():
        assert set(value)==set(s['methods'])
        for control in ['kde_original_face_fit','kde_at_nearest_noise']:
            differences[key[:-1]+(control,)].append(value[control]-value['nearest_memory'])
    comparisons=[dict(zip(['variant','design','category','condition','control'],key),paired_improvement_error_points_12=moments(values)) for key,values in differences.items()]
    write_csv(ROOT/'results/all_records.csv',rows);write_csv(ROOT/'results/all_trials.csv',trials)
    atomic(ROOT/'results/SUMMARY.json',dict(cells=cells,paired_comparisons=comparisons,uncertainty='Sample SD/SEM over50validation MC seeds;fixed trainingseed46;triple errors normalizedto12points from8actual pairs'))
    atomic(ROOT/'results/VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),rows=6000,trial_rows=57600,cells=120,seeds_per_cell=50,parent_calibration_UU_category_records_replayed=500,csv_sha256=sha(ROOT/'results/all_records.csv'),trials_sha256=sha(ROOT/'results/all_trials.csv'),summary_sha256=sha(ROOT/'results/SUMMARY.json')))

def main():
    s=verify();r=json.loads((ROOT/'RUN.json').read_text())
    assert r['registration_complete'] and r['spec_sha256']==sha(ROOT/'SPEC.json') and r['execution_id']==os.environ['GURU_RUN_ID']
    from tests import run
    run(s)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'worker',str(i)],check=True),range(2)))
    aggregate();print('NEAREST-MEMORY YIN COMPLETE;500ORIGINAL CALIBRATION UU CONTROLS EXACT',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:main()
