import datetime,fcntl,json,platform,sys
from pathlib import Path
from common import ROOT,sha,atomic,DESIGNS
from design import plan
BASE=ROOT.parent.parent
SOURCE=BASE/'operations/local-kde-alpha-zero-20261112-v2'
PARENT=BASE/'operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc'

def main():
    with (ROOT/'.registration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'SPEC.json').exists() and not (ROOT/'RUN.json').exists()
        assert sha(SOURCE/'SPEC.json')=='467904686e16b2c9a6300886a3b9daee7d21fecfd7add4265ac51851bf4e54cd'
        assert sha(PARENT/'SPEC.json')=='73fc1ccecc0600eada6568d5a6b6c1a33b4d959aa66f395a306ac8d95514b799'
        parent=json.loads((SOURCE/'SPEC.json').read_text())
        for n in ['common.py','engine.py','design.py']:assert sha(ROOT/n)==parent['files'][n],n
        external={str(SOURCE/'SPEC.json'):sha(SOURCE/'SPEC.json')}
        for n in ['SPEC.json','results/VALIDATED.json','results/all_trials.csv','results/SUMMARY_WITH_HUMANS.json','OBJECT_CATEGORY_OVERLAP.json','PACKED_HOUSES.json']:
            external[str(PARENT/n)]=sha(PARENT/n)
        validated=json.loads((PARENT/'results/VALIDATED.json').read_text());assert validated['passed'] and validated['csv_sha256']==sha(PARENT/'results/all_trials.csv')
        designs=['faces_houses','faces_houses_objects'];models=[];oldfits={}
        for m in parent['models']:
            m=dict(m);m.pop('noise_rate',None);m.pop('noise_rates',None);m['baseline_noise']={}
            done=json.loads((Path(m['cache'])/'DONE.json').read_text());assert done['model']==m['record']
            external[str(Path(m['cache'])/'DONE.json')]=sha(Path(m['cache'])/'DONE.json')
            for n,h in done['files'].items():assert sha(Path(m['cache'])/n)==h;external[str(Path(m['cache'])/n)]=h
            v=m['record']['variant'];oldfits[v]={}
            for design in designs:
                cp=PARENT/'calibration'/m['record']['model']/design/'calibration.json'
                c=json.loads(cp.read_text());assert c['spec_sha256']==sha(PARENT/'SPEC.json')
                fit=c['fits']['single_uu'];assert fit['p1']==fit['p2']
                m['baseline_noise'][design]=fit['p1'];oldfits[v][design]=fit
                external[str(cp)]=sha(cp)
            models.append(m)
        atomic(ROOT/'ORIGINAL_KDE_FITS.json',oldfits)
        atomic(ROOT/'STIMULUS_PLANS.json',{d:{str(seed):plan(d,seed) for seed in range(101,201)} for d in designs})
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json'] and p.name not in ['SPEC.json','RUN.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=str(PARENT),source_operation=str(SOURCE),models=models,designs=designs,design_definitions={d:DESIGNS[d] for d in designs},calibration_seeds=list(range(101,151)),validation_seeds=list(range(151,201)),conditions=['UU','UI','IU','II'],face_UU_target_accuracy=1-.89/12,calibration_coarse_rates=[i/100 for i in range(0,51,5)],calibration_rule='Minimum absolute error to human faceUU92.5833% on50calibration seeds;fine.005within±.05of best coarse rate;ties lower noise',calibration_targets=['faces_UU_only'],noise_application='One per-model/design fitted XOR probability,shared study/test/allorientations/allcategories;Bernoulli encoding retained',methods=['nearest_memory','kde_original_face_fit','kde_at_nearest_noise'],nearest_memory_definition='Mean10binary study fixation codes per memory;mean32binary test codes per query;full256D squaredEuclidean distance;retrieve closest amongALL40memories;choose old iff its nearest-memory distance is strictly smaller than novel;no identity labels enter scoring',kde_definition='Original float32 exp/mean/log(+1e-12),double fixation sum,max40memories;sigma2',study_fixations=10,test_fixations=32,memory_items=40,prototype_dimensions=256,distance_arithmetic='Squared difference of integer fixation-count numerators divided by common(nstudy*ntest)^2 denominator;binary input numerators/sums exact in float64;equal distances yield exact ties',expected_validation_category_rows=6000,expected_validation_trial_rows=57600,expected_cells=120,expected_parent_control_replays=500,report_error_scale='Actual counts retained;allcomparison errorpoints normalizedto12,including triple8trials/category',object_bank_scope='Same frozen64Yinobjects;16categories absent from revised trainingdiet;overlap ledger preserved;no object/human fitting',execution_platform='Local macOS arm64 CPU,torch2.5.1',files=files,external_files=external)
        import torch
        assert torch.__version__=='2.5.1'
        atomic(ROOT/'SPEC.json',spec)
        atomic(ROOT/'RUN.json',dict(registration_complete=True,execution_id='local-yin-nearest-memory-20261113-v2',spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_walltime_seconds=1800,parallel_model_workers=2,threads_per_worker=1,python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,platform=platform.platform(),scope='User-authorized nearest-memory readout Yin with two/threecategories and faceUUcalibration;no retraining or new data'))
        print((ROOT/'RUN.json').read_text())

if __name__=='__main__':main()
