import datetime,fcntl,json,subprocess
from common import ROOT,sha,atomic,verify
def main():
 with (ROOT/'.submission.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'JOBS.json').exists(),'Already registered; do not duplicate submission'
  verify();(ROOT/'logs').mkdir(exist_ok=True);created=[];jobs={'smoke':[],'production':[]}
  try:
   for mode,partition,minutes in [('smoke','gpu-shared',30),('production','shared',240)]:
    args=['sbatch','--parsable','--hold','--account=sdp159','--partition='+partition,'--cpus-per-task=4','--mem=16G','--time='+str(minutes),'--kill-on-invalid-dep=yes','--job-name=yin-nogen46-'+mode+'-1111','--output='+str(ROOT/'logs'/(mode+'-%j.log'))]
    if mode=='smoke':args+=['--gpus=1']
    else:args+=['--dependency=afterok:'+jobs['smoke'][0]]
    job=subprocess.check_output(args+[str(ROOT/'job.sbatch'),mode],universal_newlines=True).strip().split(';')[0];assert job.isdigit();created.append(job);jobs[mode].append(job)
   atomic(ROOT/'JOBS.json',dict(registration_complete=True,spec_sha256=sha(ROOT/'SPEC.json'),jobs=jobs,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Completed Delta no-generic-house ResNet18 LP/CNN training-seed46 final epoch124; inherited fixed image-bank Yin suite, seven designs, three freshly fitted noise protocols,50simulation seeds101-150; no retraining',maximum_gpu_hours=.5,maximum_cpu_core_hours=16))
   for job in created:subprocess.run(['scontrol','release',job],check=True)
  except Exception:
   for job in created:subprocess.run(['scancel',job],check=False)
   raise
  print((ROOT/'JOBS.json').read_text(),flush=True)
if __name__=='__main__':main()
