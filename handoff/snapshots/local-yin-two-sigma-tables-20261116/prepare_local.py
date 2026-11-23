import ast,datetime,fcntl,json,platform,sys
from pathlib import Path
from common import ROOT,sha,atomic,DESIGNS
from design import plan
BASE=ROOT.parent.parent
SOURCE=BASE/'operations/local-yin-new-sigma-two-noise-20261116'

def main():
    with (ROOT/'.registration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'SPEC.json').exists() and not (ROOT/'RUN.json').exists()
        assert sha(SOURCE/'SPEC.json')=='ea22b2ff2ff7968b67ae414ed19e850a1b7a57d1b5d375e9277c51923e575111'
        parent=json.loads((SOURCE/'SPEC.json').read_text())
        for n in ['common.py','design.py','engine.py','kde.py','tests.py','supervise.py']:assert sha(ROOT/n)==parent['files'][n],n
        def func(p):
            txt=p.read_text();node=next(n for n in ast.parse(txt).body if isinstance(n,ast.FunctionDef) and n.name=='fit_axis');return ast.get_source_segment(txt,node)
        assert func(ROOT/'calibration.py')==func(SOURCE/'run.py')
        external=dict(parent['external_files'])
        for n in ['SPEC.json','run.py','INDEPENDENT_VERIFICATION.json','results/CALIBRATION_SET_DIAGNOSTIC.csv','results/CALIBRATION_SET_DIAGNOSTIC.json']:
            external[str(SOURCE/n)]=sha(SOURCE/n)
        for n,h in external.items():assert sha(n)==h,n
        models=[]
        for m in parent['models']:
            m=dict(m);v=m['record']['variant'];p=SOURCE/'calibration'/(v+'.json');f=json.loads(p.read_text());assert f['spec_sha256']==sha(SOURCE/'SPEC.json') and f['sigma']==m['new_sigma']
            m['adjusted_p1']=f['p1'];m['adjusted_p2']=f['p2'];m['adjusted_calibration_file']=str(p);external[str(p)]=sha(p);models.append(m)
        designs=parent['designs']
        atomic(ROOT/'STIMULUS_PLANS.json',{d:{str(seed):plan(d,seed) for seed in parent['calibration_seeds']} for d in designs})
        files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json'] and p.name not in ['SPEC.json','RUN.json','LATEST_STATUS.json','RESUME.md']}
        spec=dict(schema=1,parent_root=parent['parent_root'],adjusted_source_operation=str(SOURCE),adjusted_source_spec_sha256=sha(SOURCE/'SPEC.json'),adjusted_calibration_set_csv=str(SOURCE/'results/CALIBRATION_SET_DIAGNOSTIC.csv'),models=models,designs=designs,design_definitions={d:DESIGNS[d] for d in designs},conditions=parent['conditions'],calibration_seeds=parent['calibration_seeds'],report_seeds=parent['calibration_seeds'],targets=parent['targets'],coarse_noise_grid=parent['coarse_noise_grid'],calibration_rule=parent['fit_rule'],noise_assignment=parent['noise_assignment'],experiments=['fixed_sigma','adjusted_sigma'],fixed_sigma=2.,adjusted_sigma_selection='Reuse frozen prior values,not reselected for this report',reporting='Same50fitting/reportseeds101-150;faceUU/IIarefitted;keepfresh201-250resultsseparate;noindependentvalidationclaim',study_fixations=10,test_fixations=32,memory_items=40,dimensions=256,expected_category_rows=3200,expected_cells=64,expected_contrasts=16,objects='20faces+20objects,transferF/Hfitunchanged;frozenbank16out-of-dietcategories;airplanesreferenceonly',files=files,external_files=external)
        import torch
        assert torch.__version__=='2.5.1'
        atomic(ROOT/'SPEC.json',spec)
        atomic(ROOT/'RUN.json',dict(registration_complete=True,execution_id='local-yin-two-sigma-tables-20261116',spec_sha256=sha(ROOT/'SPEC.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_walltime_seconds=1800,parallel_model_workers=2,threads_per_worker=1,python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,platform=platform.platform(),scope='Two requested anchored tablesets:same newer checkpoints/data,seeds/calibration method;fixedsigma2versusadjustedsigma;no training/default change'))
        print((ROOT/'RUN.json').read_text())

if __name__=='__main__':main()
