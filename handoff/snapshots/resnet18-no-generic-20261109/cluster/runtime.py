import fcntl,importlib.metadata,json,math,os,subprocess,sys
from common import sha,atomic,identity
from recipe import ROOT,SITE,SPEC,CODE,OPS,INCOMING,DATA,PYTHON,SEEDS,MAX_CHUNKS,arguments,run

def frozen():
 for name,digest in SPEC['source_files'].items():assert sha(CODE/name)==digest,'Scientific source changed: '+name
 return {'spec_sha256':sha(ROOT/'SPEC.json'),'site_sha256':sha(ROOT/'site.json'),'source_commit':SPEC['source_commit'],'source_files':SPEC['source_files'],'environment':{'python':identity(PYTHON),'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','opencv-python-headless','antialiased-cnns']}},'operations':{p.name:sha(p) for p in sorted(OPS.iterdir()) if p.suffix in ['.py','.sbatch']},'inputs':{name:sha(INCOMING/name) for name in ['photo_manifest.json','class_order.txt','CURRICULUM.json','VERIFIED.json','PHYSICAL_TOY_OVERRIDES.json','BASELINE_EXPECTED.json','COORDINATOR_PACK_JOB.json','PACK_EXPORT.json','OBJECTS_PACK_READY.json']}}
def verify(kind,variant,seed,chunk):
 reg=json.loads((ROOT/'registration.json').read_text());assert reg['registration_complete']
 match=[j for j in reg['jobs'] if (j['kind'],j['variant'],j['seed'],j['chunk'])==(kind,variant,seed,chunk)]
 assert len(match)==1 and match[0]['job_id']==os.environ['SLURM_JOB_ID'];assert reg['frozen']==frozen(),'Registered recipe or source changed'
 return reg

def verify_data():
 d=json.loads((ROOT/'DATA_VERIFIED.json').read_text());assert d['photo_manifest_sha256']==SPEC['photo_manifest_sha256'];assert d['object_pack_receipt_sha256']==sha(DATA/'OBJECTS_PACK_READY.json')
 for name,known in d['files'].items():assert identity(DATA/name)==known,'Data changed after validation: '+name
 return sha(ROOT/'DATA_VERIFIED.json')
def artifacts(variant,seed):
 import torch
 out=run(variant,seed);s=json.loads((out/'summary.json').read_text());ck=torch.load(out/'checkpoint_last.pth',map_location='cpu',weights_only=False)
 assert s['finished'] and s['config']['variant']==variant and s['config']['seed']==seed and s['config']['backbone']=='resnet18' and not s['config']['pretrained'] and s['num_classes']==2240
 assert ck['stage_idx']==10 and ck['history'][-1]['stage']==10 and ck['optimizer']['state'] and ck['rng']
 assert [h['epoch'] for h in ck['history']]==list(range(1,len(ck['history'])+1)) and all(math.isfinite(h['train_loss']) for h in ck['history'])
 return {'epochs':len(ck['history']),'checkpoint_sha256':sha(out/'checkpoint_last.pth'),'summary_sha256':sha(out/'summary.json')}
def main():
 kind,v,seed,chunk=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]);reg=verify(kind,v,seed,chunk)
 if kind=='prepare':
  subprocess.run([PYTHON,'-u',str(OPS/'prepare.py')],cwd=ROOT,check=True);verify_data();verify(kind,v,seed,chunk)
  atomic(ROOT/'PREPARED.json',{'registration_sha256':sha(ROOT/'registration.json'),'data_verified_sha256':sha(ROOT/'DATA_VERIFIED.json'),'smoke':{f'{variant}_s{s}':sha(ROOT/'smoke'/f'{variant}_s{s}'/'SMOKE_PASSED.json') for s in SEEDS for variant in ['lp','cnn']},'job_id':os.environ['SLURM_JOB_ID']});return
 assert kind=='train' and v in ['lp','cnn'] and seed in SEEDS and 1<=chunk<=MAX_CHUNKS
 prepared=json.loads((ROOT/'PREPARED.json').read_text());assert prepared['registration_sha256']==sha(ROOT/'registration.json') and prepared['data_verified_sha256']==verify_data()
 for name,digest in prepared['smoke'].items():assert sha(ROOT/'smoke'/name/'SMOKE_PASSED.json')==digest
 out=run(v,seed);out.mkdir(parents=True,exist_ok=True)
 with (out/'.training.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);provenance={'registration_sha256':sha(ROOT/'registration.json'),'prepared':prepared,'arguments':arguments(v,seed)}
  if (out/'run_provenance.json').exists():assert json.loads((out/'run_provenance.json').read_text())==provenance
  else:atomic(out/'run_provenance.json',provenance)
  if (out/'COMPLETE').exists():
   marker=json.loads((out/'COMPLETE').read_text());assert marker['provenance']==provenance and marker['artifacts']==artifacts(v,seed);print('Already complete',v,seed,flush=True);return
  atomic(out/'training_recipe.json',{'arguments':[PYTHON,'-u',*arguments(v,seed)],'variant':v,'seed':seed})
  subprocess.run([PYTHON,'-u',*arguments(v,seed)],cwd=ROOT,check=True);verify(kind,v,seed,chunk)
  atomic(out/'COMPLETE',{'provenance':provenance,'artifacts':artifacts(v,seed),'job_id':os.environ['SLURM_JOB_ID']})
  for j in reg['jobs']:
   if j['variant']==v and j['seed']==seed and j['chunk']>chunk:subprocess.run(['scontrol','update','JobId='+j['job_id'],'TimeLimit=5'],check=False)
  print('TRAINING COMPLETE',v,seed,flush=True)
if __name__=='__main__':main()
