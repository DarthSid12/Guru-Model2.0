"""Register exactly one held job before release; refuse duplicate submissions."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import datetime

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (ROOT/'.submission.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'JOBS.json').exists(), 'Job already registered; inspect it instead of resubmitting.'
    s=json.loads((ROOT/'SPEC.json').read_text())
    for rel,h in s['files'].items():assert sha(ROOT/rel)==h,rel
    (ROOT/'logs').mkdir(exist_ok=True)
    command=['sbatch','--hold','--parsable','--job-name=kanw-setA-r18-sdsc-20261106','--output='+str(ROOT/'logs/run-%j.log'),str(ROOT/'job-v2.sbatch')]
    result=subprocess.check_output(command,universal_newlines=True).strip();job=result.split(';')[0];assert job.isdigit(),result
    record={'job_id':job,'cluster':'SDSC','root':str(ROOT),'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'spec_sha256':sha(ROOT/'SPEC.json'),'submit_command':command,'models':s['models'],'maximum_gpu_hours':.5,'expected_rows':200,'status':'REGISTERED_HELD','job_script_sha256':sha(ROOT/'job-v2.sbatch'),'submit_script_sha256':sha(ROOT/'submit-v2.py'),'scheduler_recovery':'Original unsubmitted wrapper used --gres; SDSC requires --gpus=1. Scientific specification preserved unchanged.'}
    (ROOT/'JOBS.json').write_text(json.dumps(record,indent=2))
    subprocess.run(['scontrol','release',job],check=True)
    (ROOT/'RELEASED.json').write_text(json.dumps({'job_id':job,'registration_sha256':sha(ROOT/'JOBS.json'),'released_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    print(json.dumps(record,indent=2),flush=True)
