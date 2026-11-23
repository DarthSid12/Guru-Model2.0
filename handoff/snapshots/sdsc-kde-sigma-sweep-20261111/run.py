import csv,json,math,os,statistics,subprocess,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import ROOT,sha,atomic

def verify():
    s=json.loads((ROOT/'SPEC.json').read_text())
    for n,h in s['files'].items():assert sha(ROOT/n)==h,n
    for n,h in s['external_files'].items():assert sha(n)==h,n
    return s

def write_csv(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p.with_suffix('.tmp').replace(p)

def worker(index):
    import torch
    from engine import Experiment
    from design import plan
    from sigma import inputs,distances,score,reference
    torch.set_num_threads(1)
    s=verify();m=s['models'][index];variant=m['record']['variant']
    saved=json.loads((ROOT/'STIMULUS_PLANS.json').read_text())[s['design']]
    for seed in s['simulation_seeds']:assert json.loads(json.dumps(plan(s['design'],seed)))==saved[str(seed)]
    baseline={}
    for r in csv.DictReader((Path(s['noise_sweep_root'])/'results/all_trials.csv').open()):
        if r['model']==m['record']['model'] and float(r['noise_rate']) in m['noise_rates']:
            baseline[int(r['simulation_seed']),float(r['noise_rate']),r['condition'],r['category']]=int(r['correct_pairs'])
    assert len(baseline)==800
    e=Experiment(m['cache'],s['design'])
    refs={rate:reference(e,rate) for rate in m['noise_rates']}
    atomic(ROOT/'results'/(variant+'-DISTANCES.json'),refs)
    rows=[];replayed=0
    for seed in s['simulation_seeds']:
        e.draws.clear()
        for rate in m['noise_rates']:
            for condition in s['conditions']:
                p,mem,probes=inputs(e,seed,condition,rate);d=distances(mem,probes)
                baseline_scores,_=score(d,2.,'legacy_floor')
                baseline_ok=(baseline_scores.reshape(24,2)[:,0]>baseline_scores.reshape(24,2)[:,1])
                for candidate in refs[rate]['candidates']:
                    for mode in s['readouts']:
                        scores,floor_fraction=score(d,candidate['sigma'],mode)
                        scores=scores.reshape(24,2);ok=(scores[:,0]>scores[:,1]);ties=(scores[:,0]==scores[:,1])
                        for cat in ['faces','houses']:
                            idx=[i for i,t in enumerate(p['trials']) if t['category']==cat]
                            correct=int(ok[idx].sum());n_ties=int(ties[idx].sum());changed=int((ok[idx]!=baseline_ok[idx]).sum())
                            if candidate['label']=='fixed_2' and mode=='legacy_floor':
                                assert correct==baseline[seed,rate,condition,cat],(variant,seed,rate,condition,cat)
                                replayed+=1
                            rows.append(dict(model=m['record']['model'],variant=variant,checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,design=s['design'],category=cat,condition=condition,simulation_seed=seed,noise_rate=rate,sigma_label=candidate['label'],sigma=candidate['sigma'],distance_reference=refs[rate]['d_bar'],readout=mode,n_trials=12,correct_pairs=correct,errors=12-correct,correct_mask=''.join('1' if ok[i] else '0' for i in idx),ties=n_ties,changed_decisions_vs_legacy_sigma2=changed,fraction_memory_probe_fixation_densities_below_1e_12=floor_fraction))
        if seed%10==0:print('PROGRESS',variant,seed-100,'/50',flush=True)
    assert len(rows)==9600 and replayed==800
    path=ROOT/'results'/(variant+'-raw.csv');write_csv(path,rows)
    atomic(path.with_name(variant+'-DONE.json'),dict(spec_sha256=sha(ROOT/'SPEC.json'),csv_sha256=sha(path),distance_sha256=sha(ROOT/'results'/(variant+'-DISTANCES.json')),rows=len(rows),baseline_rows_reproduced_exactly=replayed,study_plans_verified=50))

def moments(v):
    sd=statistics.stdev(v)
    return dict(n=len(v),mean_errors=statistics.mean(v),sd_errors=sd,sem_errors=sd/math.sqrt(len(v)))

def aggregate():
    s=verify();rows=[];groups=defaultdict(list);paired=defaultdict(dict)
    for m in s['models']:
        v=m['record']['variant'];p=ROOT/'results'/(v+'-raw.csv');done=json.loads(p.with_name(v+'-DONE.json').read_text())
        assert done['spec_sha256']==sha(ROOT/'SPEC.json') and done['csv_sha256']==sha(p) and done['baseline_rows_reproduced_exactly']==800
        raw=list(csv.DictReader(p.open()));assert len(raw)==9600
        for r in raw:
            seed=int(r['simulation_seed']);rate=float(r['noise_rate'])
            assert r['checkpoint_sha256']==m['record']['sha256'] and rate in m['noise_rates']
            assert int(r['n_trials'])==12 and int(r['errors'])==12-int(r['correct_pairs'])
            assert len(r['correct_mask'])==12 and r['correct_mask'].count('1')==int(r['correct_pairs'])
            k=(v,r['category'],rate,r['sigma_label'],float(r['sigma']),r['readout'],r['condition'])
            groups[k].append((seed,int(r['errors']),int(r['ties']),int(r['changed_decisions_vs_legacy_sigma2']),float(r['fraction_memory_probe_fixation_densities_below_1e_12'])))
            assert r['condition'] not in paired[k[:-1],seed]
            paired[k[:-1],seed][r['condition']]=int(r['errors'])
        rows.extend(raw)
    assert len(rows)==19200 and len(groups)==384
    cells=[]
    fields=['variant','category','noise_rate','sigma_label','sigma','readout','condition']
    for key,values in groups.items():
        assert sorted(x[0] for x in values)==s['simulation_seeds']
        cells.append(dict(zip(fields,key),**moments([x[1] for x in values]),total_tied_pairs=sum(x[2] for x in values),total_changed_decisions_vs_legacy_sigma2=sum(x[3] for x in values),mean_fraction_density_below_floor=statistics.mean(x[4] for x in values)))
    contrast=defaultdict(list)
    for (key,seed),conds in paired.items():
        assert set(conds)==set(s['conditions'])
        contrast[key].append(conds['UU']-(conds['UI']+conds['IU']+conds['II'])/3)
    contrasts=[dict(zip(fields[:-1],key),**moments(v)) for key,v in contrast.items()]
    folder=ROOT/'results';write_csv(folder/'all_trials.csv',rows)
    atomic(folder/'SUMMARY.json',dict(cells=cells,UU_minus_avg_other_error_points=contrasts,uncertainty='Sample SD and Monte Carlo SEM over 50 simulation seeds; one fixed training seed46 per model; no human noise recalibration'))
    lines=['# Distance-scaled sigma diagnostic','','Errors out of12, mean ± sample SD across50 simulation seeds. Stable readout removes the additive density floor; legacy readout preserves it. Neither noise arm is recalibrated after changing sigma.','']
    for m in s['models']:
        variant=m['record']['variant']
        for rate in m['noise_rates']:
            for mode in s['readouts']:
                lines+=['## %s — noise %.2f — %s'%(variant.upper(),rate,mode),'','| Category | Sigma | UU | UI | IU | II |','|---|---:|---:|---:|---:|---:|']
                for cat in ['faces','houses']:
                    for label in s['sigma_labels']:
                        cs={c['condition']:c for c in cells if (c['variant'],c['category'],c['noise_rate'],c['readout'],c['sigma_label'])==(variant,cat,rate,mode,label)}
                        sigma=cs['UU']['sigma'];vals=' | '.join('%.2f ± %.2f'%(cs[c]['mean_errors'],cs[c]['sd_errors']) for c in s['conditions'])
                        lines.append('| %s | %.4f (%s) | %s |'%(cat,sigma,label,vals))
                lines.append('')
    (folder/'TABLES.md').write_text(chr(10).join(lines)+chr(10))
    atomic(folder/'VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),rows=19200,cells=384,seeds_per_cell=50,baseline_rows_reproduced_exactly=1600,csv_sha256=sha(folder/'all_trials.csv'),summary_sha256=sha(folder/'SUMMARY.json')))

def main():
    s=verify();j=json.loads((ROOT/'JOBS.json').read_text())
    assert j['registration_complete'] and j['job_id']==os.environ['SLURM_JOB_ID'] and j['spec_sha256']==sha(ROOT/'SPEC.json')
    from tests import run
    run(s)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'worker',str(i)],check=True),range(2)))
    aggregate();print('SIGMA SWEEP COMPLETE; BASELINE1600ROWS EXACT',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:main()
