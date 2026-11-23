import datetime,fcntl,json,subprocess
from pathlib import Path
from common import ROOT,sha,atomic
PARENT=Path('/expanse/lustre/projects/sdp159/sagrawal5/yin-house-noise-sweep-20261111')

def main():
    with (ROOT/'.submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'JOBS.json').exists() and not (ROOT/'SPEC.json').exists()
        assert sha(PARENT/'SPEC.json')=='8f1b066f47c123ab666b22fcddc4065ed160715b65373b4b681cba367721c227'
        parent=json.loads((PARENT/'SPEC.json').read_text());v=json.loads((PARENT/'results/VALIDATED.json').read_text())
        assert v['passed'] and v['csv_sha256']==sha(PARENT/'results/all_trials.csv')
        external=dict(parent['external_files'])
        for n,h in external.items():assert sha(n)==h,n
        for n in ['SPEC.json','results/VALIDATED.json','results/all_trials.csv']:
            external[str(PARENT/n)]=sha(PARENT/n)
        for n in ['common.py','design.py','engine.py','STIMULUS_PLANS.json']:
            assert sha(ROOT/n)==parent['files'][n],n
        models=[]
        for m in parent['models']:
            m=dict(m);m['noise_rates']=[0.,m['reference_noise']];models.append(m)
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.sbatch','.json'] and p.name not in ['JOBS.json','SPEC.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=parent['parent_root'],noise_sweep_root=str(PARENT),noise_sweep_spec_sha256=sha(PARENT/'SPEC.json'),models=models,design='faces_houses',simulation_seeds=list(range(101,151)),conditions=['UU','UI','IU','II'],study_items=40,study_per_category=20,trials_per_category=12,study_fixations=10,test_fixations=32,reference_seed=42,reference_orientation='U',reference_vectors='400 study-only 256bit codes after Bernoulli and added XOR; no probes or outcome optimization',reference_rule='mean Euclidean distance across all79800 unordered distinct fixation pairs',sigma_labels=['fixed_2','distance_x0.125','distance_x0.25','distance_x0.5','distance_x1.0','distance_x2.0'],sigma_rule='fixed2 plus mean_distance times[.125,.25,.5,1,2]',readouts=['legacy_floor','stable_unfloored'],readout_definition={'legacy_floor':'Original float32 exp/mean/log(+1e-12),double fixation sum,max memory','stable_unfloored':'Float64 logsumexp minus log(n_study_fixations),no additive floor,sum probe fixations,max memory'},noise_application='Same fixed added XOR rate at study/test and every orientation; no refit after sigma change; zero retains Bernoulli',expected_rows=19200,expected_cells=384,expected_baseline_replay_rows=1600,files=files,external_files=external)
        atomic(ROOT/'SPEC.json',spec);(ROOT/'logs').mkdir(exist_ok=True)
        subprocess.run(['bash','-n',str(ROOT/'job.sbatch')],check=True)
        subprocess.run(['sbatch','--test-only',str(ROOT/'job.sbatch')],check=True)
        job=subprocess.check_output(['sbatch','--parsable','--hold','--job-name=yin-kde-sigma-1111','--output='+str(ROOT/'logs/sweep-%j.log'),str(ROOT/'job.sbatch')],universal_newlines=True).strip().split(';')[0]
        assert job.isdigit()
        atomic(ROOT/'JOBS.json',dict(registration_complete=True,job_id=job,spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_cpu_core_hours=1,scope='Authorized CPU-only distance-scaled sigma exploration; frozen LP/CNN codes and study/test plans; no retraining'))
        subprocess.run(['scontrol','release',job],check=True)
        print((ROOT/'JOBS.json').read_text(),flush=True)

if __name__=='__main__':main()
