
# Support direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys, time, cProfile, pstats, io, torch
D, CKPT = sys.argv[1], sys.argv[2]
sys.argv = ['simulate_yin1969.py','--category','faces_cfdWM64','--run-dir',D,
            '--checkpoint',CKPT,'--device','cuda:0','--shuffle-items',
            '--seed','101','--noise','0.3']
from yin_tests import simulate_yin1969 as Y
# reach into main() by re-doing just its setup, then profile ONE condition.
import types
orig_rc = Y.run_condition
calls = {}
def timed_rc(*a, **k):
    t=time.time(); r=orig_rc(*a,**k); calls.setdefault('run_condition',[]).append(time.time()-t); return r
Y.run_condition = timed_rc

pr = cProfile.Profile(); pr.enable()
try:
    Y.main()
except SystemExit:
    pass
pr.disable()
s=io.StringIO(); ps=pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(28); out=s.getvalue()
print("\n".join(out.splitlines()[:46]))
print(f"\nrun_condition calls: {len(calls.get('run_condition',[]))}, "
      f"mean {sum(calls.get('run_condition',[0]))/max(len(calls.get('run_condition',[1])),1):.2f}s")
