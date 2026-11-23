import datetime,json,os,signal,subprocess,sys,time
from common import ROOT,sha,atomic

def main():
    record=json.loads((ROOT/'RUN.json').read_text())
    assert record['spec_sha256']==sha(ROOT/'SPEC.json')
    assert not (ROOT/'STARTED.json').exists()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
    (ROOT/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,GURU_RUN_ID=record['execution_id'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'logs/run.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(ROOT/'STARTED.json',dict(execution_id=record['execution_id'],pid=proc.pid,supervisor_pid=os.getpid(),started_utc=started))
        timeout=False
        try:code=proc.wait(timeout=record['maximum_walltime_seconds'])
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    atomic(ROOT/'COMPLETION.json',dict(execution_id=record['execution_id'],exit_code=code,timed_out=timeout,elapsed_seconds=time.monotonic()-t,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(ROOT/'logs/run.log')))
    print((ROOT/'COMPLETION.json').read_text())
    raise SystemExit(code or int(timeout))

if __name__=='__main__':main()
