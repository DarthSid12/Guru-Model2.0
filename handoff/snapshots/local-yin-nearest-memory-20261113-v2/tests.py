import torch
from common import ROOT,atomic
from retrieval import nearest_memory,kde_legacy,encode_inputs
from engine import Experiment

def run(s):
    torch.set_num_threads(1)
    memory=torch.tensor([[[0.,0.],[0.,0.]],[[1.,1.],[1.,1.]]])
    probes=torch.tensor([[[0.,0.],[.2,.2]],[[1.,1.],[.8,.8]],[[.5,.5],[.5,.5]]])
    score,idx,d=nearest_memory(probes,memory)
    assert idx.tolist()==[0,1,0]
    assert torch.allclose(d,torch.tensor([[.02,1.62,.5],[1.62,.02,.5]],dtype=torch.float64),atol=1e-7)
    repeat,_,_=nearest_memory(probes.repeat_interleave(2,dim=1),memory.repeat_interleave(3,dim=1))
    assert torch.allclose(repeat,score,atol=1e-12)
    rev,ri,_=nearest_memory(probes,memory.flip(0))
    assert torch.equal(rev,score) and ri[:2].tolist()==[1,0]
    swapped,_,_=nearest_memory(probes[[1,0]],memory)
    assert torch.equal(swapped,score[[1,0]])
    pilot=[]
    for m in s['models']:
        for design in s['designs']:
            e=Experiment(m['cache'],design)
            for condition in s['conditions']:
                p,mem,q=encode_inputs(e,101,condition,.3)
                a,_,_=kde_legacy(q,mem);b=e.familiarity(q,mem)
                assert torch.equal(a,b)
                sc,indices,dist=nearest_memory(q,mem)
                # Brute one prototype/query component, outside vectorized path.
                for mi,qi in [(0,0),(12,17),(39,47)]:
                    manual=sum((float(mem[mi,:,j].double().mean())-float(q[qi,:,j].double().mean()))**2 for j in range(256))
                    assert abs(manual-float(dist[mi,qi]))<1e-12
                assert sc.shape==(48,) and indices.min()>=0 and indices.max()<40
                pilot.append(dict(variant=m['record']['variant'],design=design,condition=condition,kde_scores_exact=True,nearest_distances_verified=True))
    atomic(ROOT/'results/SMOKE_PASSED.json',dict(passed=True,tests=['manual prototypes/distances','fixation replication invariance','memory permutation invariance','query permutation','full256D actual-cache distances','original KDE exact scores'],pilot=pilot))
