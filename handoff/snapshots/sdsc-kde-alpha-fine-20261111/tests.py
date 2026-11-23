import math
import torch
from engine import Experiment
from sigma import distances,score,histogram_stats,inputs
from common import atomic,ROOT

def run(s):
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(29)
    m=(torch.rand((3,4,16),generator=g)>.5).float()
    f=(torch.rand((6,5,16),generator=g)>.5).float()
    d=distances(m,f)
    brute=torch.stack([torch.stack([((m[i,:,None,None,:]-f[None,:,:,:])**2).sum(-1) for i in range(3)])])[0]
    assert torch.equal(d,brute)
    for sig in [.5,2.,20.]:
        got,_=score(d,sig,'legacy_floor')
        assert torch.equal(got,Experiment.familiarity(f,m,sig))
        stable,_=score(d,sig,'stable_unfloored')
        direct=torch.log(torch.exp(-d.double()/(2*sig*sig)).mean(1)).sum(-1).max(0).values
        assert torch.allclose(stable,direct,atol=1e-11,rtol=1e-12)
    extreme=torch.full((2,3,4,5),256.)
    stable,_=score(extreme,.1,'stable_unfloored')
    assert torch.allclose(stable,torch.full((4,),-64000.,dtype=torch.float64))
    h=[0]*257;h[0]=1;h[9]=1;h[16]=1
    st=histogram_stats(h);assert st['mean_euclidean']==7/3 and st['median_euclidean']==3
    pilots=[]
    for model in s['models']:
        e=Experiment(model['cache'],s['design'])
        for rate in model['noise_rates']:
            for condition in s['conditions']:
                p,mem,probes=inputs(e,101,condition,rate)
                a,_=score(distances(mem,probes),2.,'legacy_floor')
                b=e.familiarity(probes,mem)
                assert torch.equal(a,b)
                pilots.append(dict(variant=model['record']['variant'],noise_rate=rate,condition=condition,exact_scores_equal=True))
    atomic(ROOT/'results/SMOKE_PASSED.json',dict(passed=True,numerical_tests='Binary Hamming identity, exact legacy implementation, double direct KDE equivalence, extreme stable log scores, histogram quantiles',actual_cache_pilot=pilots))
