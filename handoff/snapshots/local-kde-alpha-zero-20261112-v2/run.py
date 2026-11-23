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
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path.with_suffix('.tmp').replace(path)

def moments(values):
    sd=statistics.stdev(values)
    return dict(n=len(values),mean_errors=statistics.mean(values),sd_errors=sd,sem_errors=sd/math.sqrt(len(values)))

def worker(index):
    import torch
    from engine import Experiment
    from design import plan
    from sigma import reference,inputs,distances,score
    torch.set_num_threads(1);s=verify();m=s['models'][index];variant=m['record']['variant'];rate=m['noise_rate']
    parent=Path(s['sigma_sweep_root']);prior={}
    for r in csv.DictReader((parent/'results/all_trials.csv').open()):
        if r['variant']==variant and float(r['noise_rate'])==rate and r['readout']=='stable_unfloored':
            prior[int(r['simulation_seed']),r['category'],r['condition'],float(r['sigma'])]=r
    assert len(prior)==2400
    e=Experiment(m['cache'],s['design']);ref=reference(e,rate)
    old=json.loads((parent/'results'/(variant+'-DISTANCES.json')).read_text())[str(rate)]
    normalized=json.loads(json.dumps(ref))
    for key in ['reference_seed','study_only','orientation','noise_rate','study_order']:
        assert normalized[key]==old[key]
    for name,stat in normalized['binary_distance_stats'].items():
        assert stat['hamming_histogram']==old['binary_distance_stats'][name]['hamming_histogram']
        assert stat['n_pairs']==old['binary_distance_stats'][name]['n_pairs']
        for field,value in stat.items():
            if field not in ['hamming_histogram','n_pairs']:
                assert math.isclose(value,old['binary_distance_stats'][name][field],abs_tol=1e-10,rel_tol=1e-12)
    for name,stat in normalized['raw_sigmoid_distance_stats'].items():
        assert stat['n_pairs']==old['raw_sigmoid_distance_stats'][name]['n_pairs']
        for field in ['mean_euclidean','sd_euclidean']:
            assert math.isclose(stat[field],old['raw_sigmoid_distance_stats'][name][field],abs_tol=1e-10,rel_tol=1e-12)
    assert math.isclose(normalized['d_bar'],old['d_bar'],abs_tol=1e-10,rel_tol=1e-12)
    # Keep exact stored parent floats after verifying exact codes and distributions.
    atomic(ROOT/'results'/(variant+'-RECOMPUTED_LOCAL_REFERENCE.json'),normalized)
    ref=old
    atomic(ROOT/'results'/(variant+'-REFERENCE_COMPATIBILITY.json'),dict(binary_histograms_exact=True,all_distance_moments_within_1e_10=True,stored_parent_reference_reused_exactly=True,reference_seed=42,noise_rate=rate))
    candidates=[dict(label='fixed_2',sigma=2.,alpha=None)]+[dict(label='alpha_'+str(a),sigma=a*ref['d_bar'],alpha=a) for a in s['alphas']]
    ref['candidates']=candidates;atomic(ROOT/'results'/(variant+'-DISTANCES.json'),ref)
    saved=json.loads((ROOT/'STIMULUS_PLANS.json').read_text())[s['design']]
    rows=[];replayed=0;overlap=0
    for seed in s['simulation_seeds']:
        e.draws.clear();p=plan(s['design'],seed)
        if seed in s['development_seeds']:assert json.loads(json.dumps(p))==saved[str(seed)]
        split='development' if seed in s['development_seeds'] else 'validation'
        for condition in s['conditions']:
            p,mem,probes=inputs(e,seed,condition,rate);d=distances(mem,probes)
            base,_=score(d,2.,'legacy_floor');baseok=(base.reshape(24,2)[:,0]>base.reshape(24,2)[:,1])
            for cand in candidates:
                scores,floor=score(d,cand['sigma'],'stable_unfloored');scores=scores.reshape(24,2)
                ok=scores[:,0]>scores[:,1];ties=scores[:,0]==scores[:,1]
                for cat in ['faces','houses']:
                    idx=[i for i,t in enumerate(p['trials']) if t['category']==cat]
                    mask=''.join('1' if ok[i] else '0' for i in idx);correct=mask.count('1')
                    if split=='development' and (seed,cat,condition,cand['sigma']) in prior:
                        r=prior[seed,cat,condition,cand['sigma']]
                        assert correct==int(r['correct_pairs']) and mask==r['correct_mask']
                        overlap+=1
                        if cand['label']=='fixed_2':replayed+=1
                    rows.append(dict(model=m['record']['model'],variant=variant,checkpoint_sha256=m['record']['sha256'],training_seed=46,epoch=124,design=s['design'],category=cat,condition=condition,simulation_seed=seed,split=split,noise_rate=rate,sigma_label=cand['label'],sigma=cand['sigma'],alpha=cand['alpha'],distance_reference=ref['d_bar'],readout='stable_unfloored',n_trials=12,correct_pairs=correct,errors=12-correct,correct_mask=mask,ties=int(ties[idx].sum()),changed_decisions_vs_legacy_sigma2=int((ok[idx]!=baseok[idx]).sum()),fraction_memory_probe_fixation_densities_below_1e_12=floor))
        if seed%10==0:print('PROGRESS',variant,seed-100,'/100',flush=True)
    assert len(rows)==19200 and replayed==400 and overlap==2400
    path=ROOT/'results'/(variant+'-raw.csv');write_csv(path,rows)
    atomic(path.with_name(variant+'-DONE.json'),dict(spec_sha256=sha(ROOT/'SPEC.json'),csv_sha256=sha(path),distance_sha256=sha(ROOT/'results'/(variant+'-DISTANCES.json')),rows=len(rows),legacy_control_rows_replayed=replayed,all_overlapping_parent_rows_and_masks_replayed=overlap,development_plans_verified=50))

def aggregate():
    s=verify();rows=[];groups=defaultdict(list);paired=defaultdict(dict)
    fields=['variant','category','noise_rate','sigma_label','sigma','split','condition']
    for m in s['models']:
        v=m['record']['variant'];path=ROOT/'results'/(v+'-raw.csv');done=json.loads(path.with_name(v+'-DONE.json').read_text())
        assert done['spec_sha256']==sha(ROOT/'SPEC.json') and done['csv_sha256']==sha(path)
        assert done['legacy_control_rows_replayed']==400 and done['all_overlapping_parent_rows_and_masks_replayed']==2400
        selected=list(csv.DictReader(path.open()));assert len(selected)==19200
        for r in selected:
            assert r['checkpoint_sha256']==m['record']['sha256'] and float(r['noise_rate'])==m['noise_rate']
            assert int(r['errors'])==12-int(r['correct_pairs']) and r['correct_mask'].count('1')==int(r['correct_pairs'])
            seed=int(r['simulation_seed']);key=(v,r['category'],float(r['noise_rate']),r['sigma_label'],float(r['sigma']),r['split'],r['condition'])
            groups[key].append((seed,int(r['errors'])))
            assert r['condition'] not in paired[key[:-1],seed]
            paired[key[:-1],seed][r['condition']]=int(r['errors'])
        rows.extend(selected)
    assert len(rows)==38400 and len(groups)==768
    cells=[]
    for key,values in groups.items():
        expected=s['development_seeds'] if key[-2]=='development' else s['validation_seeds']
        assert sorted(seed for seed,x in values)==expected
        cells.append(dict(zip(fields,key),**moments([x for seed,x in values])))
    contrasts=defaultdict(list)
    for (key,seed),d in paired.items():
        assert set(d)==set(s['conditions']);contrasts[key].append(d['UU']-(d['UI']+d['IU']+d['II'])/3)
    orientation_contrasts=[dict(zip(fields[:-1],key),**moments(values)) for key,values in contrasts.items()]
    # Winner depends only on development house UU; no validation values enter.
    selections=[]
    for m in s['models']:
        v=m['record']['variant']
        candidates=[c for c in cells if c['variant']==v and c['category']=='houses' and c['split']=='development' and c['condition']=='UU' and c['sigma_label']!='fixed_2']
        winner=min(candidates,key=lambda c:(c['mean_errors'],float(c['sigma_label'].removeprefix('alpha_'))))
        chosen=winner['sigma_label'];changes=[]
        for cat in ['faces','houses']:
            for cond in s['conditions']:
                base={seed:x for seed,x in groups[(v,cat,m['noise_rate'],'fixed_2',2.,'validation',cond)]}
                new={seed:x for seed,x in groups[(v,cat,m['noise_rate'],chosen,winner['sigma'],'validation',cond)]}
                changes.append(dict(category=cat,condition=cond,**moments([base[seed]-new[seed] for seed in s['validation_seeds']])))
        selections.append(dict(variant=v,criterion=s['selection_rule'],alpha=float(chosen.removeprefix('alpha_')),sigma=winner['sigma'],sigma_label=chosen,development_house_UU=winner,validation_paired_improvement_errors=changes))
    folder=ROOT/'results';write_csv(folder/'all_trials.csv',rows)
    atomic(folder/'DEVELOPMENT_SELECTIONS.json',dict(selection_uses_only_development=True,selections=selections,validation_is_new_simulation_seeds_not_new_stimuli=True))
    atomic(folder/'SUMMARY.json',dict(cells=cells,UU_minus_avg_other_error_points=orientation_contrasts,uncertainty='Sample SD/SEM across50simulation seeds within each split; fixed trainingseed46 and same64-photo banks; zero added XOR noise; stochastic Bernoulli encoding remains active'))
    atomic(folder/'VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),rows=38400,cells=768,seeds_per_cell=50,legacy_control_rows_replayed=800,overlapping_parent_rows_and_masks_replayed=4800,csv_sha256=sha(folder/'all_trials.csv'),summary_sha256=sha(folder/'SUMMARY.json'),selection_sha256=sha(folder/'DEVELOPMENT_SELECTIONS.json')))

def main():
    s=verify();j=json.loads((ROOT/'RUN.json').read_text())
    assert j['registration_complete'] and j['spec_sha256']==sha(ROOT/'SPEC.json') and j['execution_id']==os.environ['GURU_RUN_ID']
    from tests import run
    run(s)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'worker',str(i)],check=True),range(2)))
    aggregate();print('ZERO-NOISE ALPHA SWEEP COMPLETE;800LEGACY CONTROLS AND4800OVERLAP MASKS EXACT',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:main()
