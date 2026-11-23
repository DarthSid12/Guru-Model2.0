import datetime,fcntl,json,subprocess
from pathlib import Path
from common import ROOT,sha,atomic
PARENT=Path('/expanse/lustre/projects/sdp159/sagrawal5/yin-kde-sigma-sweep-20261111')

def main():
    with (ROOT/'.submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'SPEC.json').exists() and not (ROOT/'JOBS.json').exists()
        assert sha(PARENT/'SPEC.json')=='957700c77e1e1f2e6d1639ee5dac92cbd8367c08aec0628cb3056bdb50e627c8'
        parent=json.loads((PARENT/'SPEC.json').read_text());v=json.loads((PARENT/'results/VALIDATED.json').read_text())
        assert v['passed'] and v['csv_sha256']==sha(PARENT/'results/all_trials.csv')
        external=dict(parent['external_files'])
        for p,h in external.items():assert sha(p)==h,p
        for name in ['SPEC.json','results/VALIDATED.json','results/all_trials.csv','results/lp-DISTANCES.json','results/cnn-DISTANCES.json']:
            external[str(PARENT/name)]=sha(PARENT/name)
        for name in ['common.py','design.py','engine.py','sigma.py','tests.py','STIMULUS_PLANS.json']:
            assert sha(ROOT/name)==parent['files'][name],name
        models=[]
        for m in parent['models']:
            m=dict(m);m['noise_rates']=[m['reference_noise']];models.append(m)
        alphas=[.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.6,.75,1.,1.5,2.,3.,4.]
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.sbatch'] and p.name not in ['SPEC.json','JOBS.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=parent['parent_root'],noise_sweep_root=parent['noise_sweep_root'],sigma_sweep_root=str(PARENT),sigma_sweep_spec_sha256=sha(PARENT/'SPEC.json'),models=models,design='faces_houses',simulation_seeds=list(range(101,201)),development_seeds=list(range(101,151)),validation_seeds=list(range(151,201)),conditions=['UU','UI','IU','II'],study_items=40,study_per_category=20,trials_per_category=12,study_fixations=10,test_fixations=32,reference_seed=42,reference_rule=parent['reference_rule'],reference_vectors=parent['reference_vectors'],reference_orientation='U',alphas=alphas,sigma_labels=['fixed_2']+['alpha_'+str(a) for a in alphas],sigma_rule='fixed2 plus alpha times exact parent study-only mean binary-code distance; same across categories/orientations',readout='stable_unfloored',readout_definition=parent['readout_definition']['stable_unfloored'],noise_application=parent['noise_application'],selection_rule='Minimum development mean house UU errors among alpha candidates; ties choose smaller alpha; no validation reselection',validation_scope='New50simulation seeds on same64-photo banks; not independent stimuli or training replications',expected_rows=33600,expected_cells=672,expected_legacy_control_replays=800,expected_overlapping_parent_rows=4800,files=files,external_files=external)
        atomic(ROOT/'SPEC.json',spec);(ROOT/'logs').mkdir(exist_ok=True)
        subprocess.run(['bash','-n',str(ROOT/'job.sbatch')],check=True)
        subprocess.run(['sbatch','--test-only',str(ROOT/'job.sbatch')],check=True)
        job=subprocess.check_output(['sbatch','--parsable','--hold','--job-name=yin-alpha-fine-1111','--output='+str(ROOT/'logs/sweep-%j.log'),str(ROOT/'job.sbatch')],universal_newlines=True).strip().split(';')[0];assert job.isdigit()
        atomic(ROOT/'JOBS.json',dict(registration_complete=True,job_id=job,spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_cpu_core_hours=1,scope='Authorized finer20point alpha exploration with unchanged noise, frozen LP/CNN codes and new simulation-seed validation; no retraining'))
        subprocess.run(['scontrol','release',job],check=True)
        print((ROOT/'JOBS.json').read_text(),flush=True)

if __name__=='__main__':main()
