import datetime,fcntl,json,platform,sys
from pathlib import Path
from common import ROOT,sha,atomic,DESIGNS
from design import plan
BASE=ROOT.parent.parent
SOURCE=BASE/'operations/local-kde-alpha-zero-20261112-v2'
PARENT=BASE/'operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc'
RANK=BASE/'output/kde-sigma-human-comparison-20261116/RANKING.json'

def main():
    with (ROOT/'.registration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'SPEC.json').exists() and not (ROOT/'RUN.json').exists()
        assert sha(PARENT/'SPEC.json')=='73fc1ccecc0600eada6568d5a6b6c1a33b4d959aa66f395a306ac8d95514b799'
        parent=json.loads((SOURCE/'SPEC.json').read_text());rank=json.loads(RANK.read_text())
        for n in ['common.py','design.py','engine.py']:assert sha(ROOT/n)==parent['files'][n],n
        external={str(RANK):sha(RANK),str(SOURCE/'SPEC.json'):sha(SOURCE/'SPEC.json')}
        for n in ['SPEC.json','results/VALIDATED.json','results/all_trials.csv','results/SUMMARY_WITH_HUMANS.json','OBJECT_CATEGORY_OVERLAP.json','PACKED_HOUSES.json']:external[str(PARENT/n)]=sha(PARENT/n)
        pv=json.loads((PARENT/'results/VALIDATED.json').read_text());assert pv['passed'] and pv['csv_sha256']==sha(PARENT/'results/all_trials.csv')
        models=[]
        for m in parent['models']:
            m=dict(m);v=m['record']['variant'];m.pop('noise_rate',None);m.pop('noise_rates',None)
            m['new_sigma']=rank['winners'][v]['sigma'];m['sigma_selection_alpha']=rank['winners'][v]['alpha'];m['sigma_reference_noise']=rank['winners'][v]['noise_rate']
            assert m['new_sigma']=={'lp':4.509564323062932,'cnn':6.753987849071428}[v]
            done=json.loads((Path(m['cache'])/'DONE.json').read_text());assert done['model']==m['record']
            external[str(Path(m['cache'])/'DONE.json')]=sha(Path(m['cache'])/'DONE.json')
            for n,h in done['files'].items():assert sha(Path(m['cache'])/n)==h;external[str(Path(m['cache'])/n)]=h
            cp=PARENT/'calibration'/m['record']['model']/'faces_houses/calibration.json';c=json.loads(cp.read_text());assert c['spec_sha256']==sha(PARENT/'SPEC.json')
            fit=c['fits']['study_only_fitted'];m['original_two_noise']=dict(p1=fit['p1'],p2=fit['p2'],calibration_file=str(cp),calibration_sha256=sha(cp));external[str(cp)]=sha(cp)
            models.append(m)
        designs=['faces_houses','faces_objects']
        atomic(ROOT/'STIMULUS_PLANS.json',{d:{str(seed):plan(d,seed) for seed in list(range(101,151))+list(range(201,251))} for d in designs})
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json'] and p.name not in ['SPEC.json','RUN.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=str(PARENT),source_operation=str(SOURCE),models=models,designs=designs,design_definitions={d:DESIGNS[d] for d in designs},calibration_design='faces_houses',calibration_seeds=list(range(101,151)),validation_seeds=list(range(201,251)),sigma_selection_seeds=list(range(101,201)),targets={'UU':1-.89/12,'II':1-4.35/12},conditions=['UU','UI','IU','II'],coarse_noise_grid=[i/100 for i in range(0,51,5)],fit_rule='FaceUUfitp1sharedstudy/test,thenfaceIIfitp2studywithtestp1fixed;coarse.05,fine.005within±.05,ultra.001within±.005;minabserror,tieslowernoise',noise_protocol='study_only_fitted',noise_assignment='UU/UI(p1,p1),IU/II(p2,p1),study/test',object_noise_transfer='Face/housedesignfittransferredunchangedto20faces+20objects;noobjectanchor',methods=['new_sigma_two_noise','sigma2_two_noise_originalfit','sigma2_two_noise_matchedfit','new_sigma_single_p1'],KDE='Original float32 exp/mean/log(+1e-12),doublefixationsum,max40memories;no readoutchange',study_fixations=10,test_fixations=32,memory_items=40,dimensions=256,expected_category_rows=6400,expected_trial_rows=76800,expected_condition_cells=128,expected_parent_control_replays=400,object_bank_scope='Frozen64wordbankimages;16categoriesabsentfromrevisedtrainingdiet;historicalairplanesreferenceonly',execution_platform='LocalMacCPU,torch2.5.1',files=files,external_files=external)
        assert set(spec['validation_seeds']).isdisjoint(spec['sigma_selection_seeds'])
        import torch
        assert torch.__version__=='2.5.1'
        atomic(ROOT/'SPEC.json',spec)
        atomic(ROOT/'RUN.json',dict(registration_complete=True,execution_id='local-yin-new-sigma-two-noise-20261116',spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_walltime_seconds=1800,parallel_model_workers=2,threads_per_worker=1,python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,platform=platform.platform(),scope='User-authorized newer LP/CNN+selected newKDEsigmas+two faceanchors;fixed absolute sigma;fresh201-250evaluation;no training or activebaseline change'))
        print((ROOT/'RUN.json').read_text())

if __name__=='__main__':main()
