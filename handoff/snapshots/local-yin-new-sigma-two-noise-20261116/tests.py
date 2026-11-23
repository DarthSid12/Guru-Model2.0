import torch
from common import ROOT,atomic
from engine import Experiment
from kde import inputs,score,noise_assignment

def run(s):
    torch.set_num_threads(1);pilot=[]
    for c in s['conditions']:
        assert noise_assignment(c,.2,.4)==((.2 if c[0]=='U' else .4),.2)
    for m in s['models']:
        for design in s['designs']:
            e=Experiment(m['cache'],design)
            for condition in s['conditions']:
                sn,tn=noise_assignment(condition,.31,.42);p,mem,q=inputs(e,101,condition,sn,tn)
                for sig in [2.,m['new_sigma']]:
                    a,_=score(q,mem,sig);b=e.familiarity(q,mem,sig)
                    assert torch.equal(a,b)
                assert mem.shape==(40,10,256) and q.shape==(48,32,256)
                pilot.append(dict(variant=m['record']['variant'],design=design,condition=condition,exact_original_scores_for_both_sigmas=True))
    atomic(ROOT/'results/SMOKE_PASSED.json',dict(passed=True,study_only_noise_rule_verified=True,pilot=pilot))
