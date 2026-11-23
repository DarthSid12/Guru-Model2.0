import fcntl,json,subprocess
from recipe import ROOT,SITE,SPEC,OPS,PYTHON,MAX_CHUNKS,CHUNK_HOURS,SEEDS,run,arguments
from common import sha,atomic
from runtime import frozen

def main():
 with (ROOT/'.submission.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (ROOT/'registration.json').exists(),'Already registered; do not duplicate jobs'
  assert all(not run(v,s).exists() for s in SEEDS for v in ['lp','cnn'])
  subprocess.run(['bash','-n',str(OPS/'train.sbatch')],check=True)
  pack=json.loads((ROOT/'incoming/COORDINATOR_PACK_JOB.json').read_text())
  assert pack['job_id']==SPEC['reused_pack_coordinator_job']
  r={'schema':1,'registration_complete':False,'seeds':list(SEEDS),'variants':['lp','cnn'],'backbone':'resnet18','pretrained':False,'fresh_initialization':True,'account':SITE['account'],'partition':SITE['partition'],'maximum_training_gpu_hours':2*MAX_CHUNKS*CHUNK_HOURS,'maximum_preparation_gpu_hours':1,'source_commit':SPEC['source_commit'],'object_source_manifest_sha256':SPEC['photo_manifest_sha256'],'coordinator_pack_job_id':pack['job_id'],'frozen':frozen(),'jobs':[],'commands':{v:[PYTHON,'-u',*arguments(v,SITE['seed'])] for v in ['lp','cnn']},'authorization':'User requested LP/CNN ResNet18 with updated object bank on both clusters; explicitly selected seed45 on Anvil and seed46 on Delta; remove generic-house data and label; preserve face/object/house identities, data splits and all other settings; all jobs held pending diet review.'}
  jobs=[('prepare','both',0,0)]+[('train',v,SITE['seed'],c) for v in ['lp','cnn'] for c in [1,2]];prepare=None;previous={}
  try:
   for kind,v,seed,chunk in jobs:
    dep=None
    if kind=='train':dep='afterok:'+prepare+(',afterany:'+previous[v] if v in previous else '')
    cmd=['sbatch','--parsable','--hold','--account='+SITE['account'],'--partition='+SITE['partition'],'--time='+('01:00:00' if kind=='prepare' else f'{CHUNK_HOURS:02d}:00:00'),'--job-name='+f'nogen-r18-s{seed}-{v}-c{chunk}-1109','--kill-on-invalid-dep=yes','--output='+str(ROOT/f'logs/{kind}-{v}-s{seed}-c{chunk}-%j.log')]
    if dep:cmd.append('--dependency='+dep)
    job=subprocess.check_output(cmd+[str(OPS/'train.sbatch'),kind,v,str(seed),str(chunk)],universal_newlines=True).strip().split(';')[0];assert job.isdigit()
    r['jobs'].append({'kind':kind,'variant':v,'seed':seed,'chunk':chunk,'job_id':job,'dependency':dep});atomic(ROOT/'registration.json',r)
    if kind=='prepare':prepare=job
    else:previous[v]=job
   r['registration_complete']=True;atomic(ROOT/'registration.json',r)
  except Exception:
   if r['jobs']:subprocess.run(['scancel',*[j['job_id'] for j in r['jobs']]],check=False)
   raise
  atomic(ROOT/'HELD_FOR_DIET_REVIEW.json',{'registration_sha256':sha(ROOT/'registration.json'),'jobs':r['jobs'],'reason':'User requested diet preview before training starts','all_jobs_held':True})
  print(json.dumps({'jobs':r['jobs'],'registration_sha256':sha(ROOT/'registration.json'),'all_jobs_held_for_diet_review':True}),flush=True)
if __name__=='__main__':main()
