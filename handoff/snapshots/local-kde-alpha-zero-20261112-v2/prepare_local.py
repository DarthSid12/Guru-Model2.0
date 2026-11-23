import datetime,fcntl,json,platform,sys
from pathlib import Path
from common import ROOT,sha,atomic
BASE=ROOT.parent.parent
PARENT=BASE/'operations/sdsc-kde-sigma-sweep-20261111/completed-sdsc'

def main():
    with (ROOT/'.registration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'SPEC.json').exists() and not (ROOT/'RUN.json').exists()
        assert sha(PARENT/'SPEC.json')=='957700c77e1e1f2e6d1639ee5dac92cbd8367c08aec0628cb3056bdb50e627c8'
        parent=json.loads((PARENT/'SPEC.json').read_text());v=json.loads((PARENT/'results/VALIDATED.json').read_text())
        assert v['passed'] and v['csv_sha256']==sha(PARENT/'results/all_trials.csv')
        maps=[(str(PARENT),PARENT),('/expanse/lustre/projects/sdp159/sagrawal5/yin-kde-sigma-sweep-20261111',PARENT),(parent['noise_sweep_root'],BASE/'operations/sdsc-house-noise-sweep-20261111/completed-sdsc'),(parent['parent_root'],BASE/'operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc')]
        def local(p):
            matches=[new/Path(p).relative_to(old) for old,new in maps if Path(p).is_relative_to(old)]
            assert len(matches)==1,p
            return matches[0]
        external={str(local(p)):h for p,h in parent['external_files'].items()}
        original_paths={str(local(p)):p for p in parent['external_files']}
        for p,h in external.items():assert sha(p)==h,p
        for n in ['SPEC.json','results/VALIDATED.json','results/all_trials.csv','results/lp-DISTANCES.json','results/cnn-DISTANCES.json']:
            external[str(PARENT/n)]=sha(PARENT/n)
        for n in ['common.py','design.py','engine.py','sigma.py','tests.py','STIMULUS_PLANS.json']:
            assert sha(ROOT/n)==parent['files'][n],n
        models=[]
        for m in parent['models']:
            m=dict(m);m['source_cache']=m['cache'];m['cache']=str(local(m['cache']));m['noise_rate']=0.;m['noise_rates']=[0.];models.append(m)
        alphas=[.005,.01,.025,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.6,.75,1.,1.5,2.,3.,4.]
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json'] and p.name not in ['SPEC.json','RUN.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=str(local(parent['parent_root'])),noise_sweep_root=str(local(parent['noise_sweep_root'])),sigma_sweep_root=str(PARENT),sigma_sweep_spec_sha256=sha(PARENT/'SPEC.json'),execution_platform='macOS arm64 CPU; torch2.5.1',original_external_paths=original_paths,models=models,design='faces_houses',simulation_seeds=list(range(101,201)),development_seeds=list(range(101,151)),validation_seeds=list(range(151,201)),conditions=['UU','UI','IU','II'],study_items=40,study_per_category=20,trials_per_category=12,study_fixations=10,test_fixations=32,reference_seed=42,reference_rule=parent['reference_rule'],reference_vectors=parent['reference_vectors'],reference_orientation='U',reference_compatibility='Exact binary histograms; all binary/continuous moments within1e-10; stored original parent reference floats reused exactly after verification',alphas=alphas,sigma_labels=['fixed_2']+['alpha_'+str(a) for a in alphas],sigma_rule='fixed2 plus alpha times exact parent study-only mean binary-code distance; same across categories/orientations',readout='stable_unfloored',readout_definition=parent['readout_definition']['stable_unfloored'],noise_application='Zero added XOR noise during study and test, all categories/orientations; Bernoulli remains stochastic',selection_rule='Minimum development mean house UU errors among alpha candidates; ties choose smaller alpha; no validation reselection',validation_scope='50new simulation seeds on same64-photo banks; not independent stimuli or training replications',expected_rows=38400,expected_cells=768,expected_legacy_control_replays=800,expected_overlapping_parent_rows=4800,files=files,external_files=external)
        import torch
        assert torch.__version__=='2.5.1' and platform.machine()=='arm64'
        atomic(ROOT/'SPEC.json',spec)
        atomic(ROOT/'RUN.json',dict(registration_complete=True,execution_id='local-alpha-zero-20261112-v2',spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_walltime_seconds=1800,parallel_model_workers=2,threads_per_worker=1,python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,platform=platform.platform(),scope='User-authorized23point alpha sweep at noise0; local CPU cached-code readout; no retraining or GPU use'))
        print((ROOT/'RUN.json').read_text())

if __name__=='__main__':main()
