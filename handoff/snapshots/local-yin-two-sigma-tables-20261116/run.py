import csv,json,math,statistics as st,subprocess,sys,os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

def worker(index):
    import torch
    from calibration import fit_axis
    from engine import Experiment
    from kde import inputs,score,noise_assignment
    from design import plan
    torch.set_num_threads(1);s=verify();m=s['models'][index];v=m['record']['variant'];e=Experiment(m['cache'],'faces_houses')
    u=fit_axis(e,s,2.,'UU',s['targets']['UU']);p1=u['selected_noise'];i=fit_axis(e,s,2.,'II',s['targets']['II'],p1);p2=i['selected_noise']
    fit=dict(spec_sha256=sha(ROOT/'SPEC.json'),experiment='fixed_sigma',sigma=2.,p1=p1,p2=p2,UU_fit=u,II_fit=i,calibration_seeds=s['calibration_seeds'])
    atomic(ROOT/'calibration'/(v+'-fixed_sigma.json'),fit);print('FIXED-SIGMA CALIBRATED',v,p1,p2,u['achieved'],i['achieved'],flush=True)
    imported=json.loads(Path(m['adjusted_calibration_file']).read_text());assert imported['p1']==m['adjusted_p1'] and imported['p2']==m['adjusted_p2'] and imported['sigma']==m['new_sigma']
    atomic(ROOT/'calibration'/(v+'-adjusted_sigma.json'),dict(source_file=m['adjusted_calibration_file'],source_sha256=sha(m['adjusted_calibration_file']),source_spec_sha256=s['adjusted_source_spec_sha256'],fit=imported))
    original=list(csv.DictReader(Path(s['adjusted_calibration_set_csv']).open()));old={(r['design'],int(r['simulation_seed']),r['condition'],r['category']):r for r in original if r['variant']==v}
    assert len(old)==800
    rows=[]
    for design in s['designs']:
        e=Experiment(m['cache'],design)
        for seed in s['calibration_seeds']:
            e.draws.clear()
            for condition in s['conditions']:
                sn,tn=noise_assignment(condition,p1,p2);p,mem,q=inputs(e,seed,condition,sn,tn);values,_=score(q,mem,2.)
                correct=values.reshape(24,2)[:,0]>values.reshape(24,2)[:,1]
                for cat in DESIGNS[design]:
                    mask=''.join('1' if correct[j] else '0' for j,t in enumerate(p['trials']) if t['category']==cat);count=mask.count('1')
                    rows.append(dict(variant=v,model=m['record']['model'],checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,experiment='fixed_sigma',sigma=2.,p1=p1,p2=p2,study_noise=sn,test_noise=tn,design=design,category=cat,condition=condition,simulation_seed=seed,n_trials=12,correct_pairs=count,errors=12-count,accuracy_pct=100*count/12,correct_mask=mask,record_source='Matched-refit sigma2 experiment'))
                    r=old[design,seed,condition,cat];a,b=m['adjusted_p1'],m['adjusted_p2'];asn,atn=noise_assignment(condition,a,b)
                    assert float(r['p1'])==a and float(r['p2'])==b and float(r['sigma'])==m['new_sigma'] and int(r['n_trials'])==12
                    rows.append(dict(variant=v,model=m['record']['model'],checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,experiment='adjusted_sigma',sigma=m['new_sigma'],p1=a,p2=b,study_noise=asn,test_noise=atn,design=design,category=cat,condition=condition,simulation_seed=seed,n_trials=12,correct_pairs=int(r['correct_pairs']),errors=int(r['errors']),accuracy_pct=100*int(r['correct_pairs'])/12,correct_mask='',record_source=s['adjusted_calibration_set_csv']))
            if seed%10==0:print('PROGRESS',v,design,seed-100,'/50',flush=True)
    assert len(rows)==1600
    write_csv(ROOT/'results'/(v+'-raw.csv'),rows)
    atomic(ROOT/'results'/(v+'-DONE.json'),dict(spec_sha256=sha(ROOT/'SPEC.json'),rows=len(rows),adjusted_records_imported=800,csv_sha256=sha(ROOT/'results'/(v+'-raw.csv')),fixed_fit_sha256=sha(ROOT/'calibration'/(v+'-fixed_sigma.json'))))

def moments(v):
    sd=st.stdev(v);return dict(n=len(v),mean=st.mean(v),sd=sd,sem=sd/math.sqrt(len(v)))

def aggregate():
    s=verify();rows=[];groups=defaultdict(list);paired=defaultdict(dict)
    for m in s['models']:
        v=m['record']['variant'];p=ROOT/'results'/(v+'-raw.csv');d=json.loads(p.with_name(v+'-DONE.json').read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json') and d['csv_sha256']==sha(p)
        rs=list(csv.DictReader(p.open()));assert len(rs)==1600;rows.extend(rs)
        for r in rs:
            assert int(r['errors'])+int(r['correct_pairs'])==12 and int(r['n_trials'])==12
            key=(r['experiment'],v,r['design'],r['category'],r['condition']);groups[key].append(r)
            pk=(r['experiment'],v,r['design'],r['category'],int(r['simulation_seed']));assert r['condition'] not in paired[pk];paired[pk][r['condition']]=int(r['errors'])
    assert len(rows)==3200 and len(groups)==64
    cells=[]
    for key,rs in groups.items():
        assert sorted(int(r['simulation_seed']) for r in rs)==s['calibration_seeds']
        vals=[int(r['errors']) for r in rs];cells.append(dict(zip(['experiment','variant','design','category','condition'],key),sigma=float(rs[0]['sigma']),p1=float(rs[0]['p1']),p2=float(rs[0]['p2']),errors=moments(vals),accuracy_pct=moments([100*(1-x/12) for x in vals])))
    contrast=defaultdict(list)
    for key,vals in paired.items():
        assert set(vals)==set(s['conditions']);contrast[key[:-1]].append(vals['UU']-(vals['UI']+vals['IU']+vals['II'])/3)
    contrasts=[dict(zip(['experiment','variant','design','category'],key),error_points=moments(vals),accuracy_percentage_points=moments([-100*x/12 for x in vals])) for key,vals in contrast.items()]
    write_csv(ROOT/'results/all_records.csv',rows)
    atomic(ROOT/'results/SUMMARY.json',dict(cells=cells,contrasts=contrasts,reporting='Same50calibration/reportseeds101-150;faceUU/IIarefitted;notindependentvalidation;sampleSD/SEM acrosssimulationseeds'))
    atomic(ROOT/'results/VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),rows=3200,cells=64,contrasts=16,simulation_seeds_per_cell=50,csv_sha256=sha(ROOT/'results/all_records.csv'),summary_sha256=sha(ROOT/'results/SUMMARY.json')))

def main():
    s=verify();r=json.loads((ROOT/'RUN.json').read_text());assert r['registration_complete'] and r['spec_sha256']==sha(ROOT/'SPEC.json') and r['execution_id']==os.environ['GURU_RUN_ID']
    from tests import run
    run(s)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'worker',str(i)],check=True),range(2)))
    aggregate();print('TWO-SIGMA ANCHORED TABLES COMPLETE',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:main()
