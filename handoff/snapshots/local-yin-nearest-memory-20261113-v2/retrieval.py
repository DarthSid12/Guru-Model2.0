"""Nearest-prototype readout. No stimulus labels enter either scoring function."""
import torch

def nearest_memory(probes,memory):
    # For binary codes these are integer sums, represented exactly in float64.
    # (sum_study/nstudy - sum_test/ntest)^2 has one shared denominator.
    nstudy=memory.shape[1];ntest=probes.shape[1]
    numerators=ntest*memory.double().sum(dim=1)[:,None,:]-nstudy*probes.double().sum(dim=1)[None,:,:]
    d=numerators.square().sum(dim=-1)/((nstudy*ntest)**2)
    best,index=d.min(dim=0)
    assert torch.isfinite(best).all()
    return -best,index,d

def kde_legacy(probes,memory,sigma=2.):
    nmem,nfix,width=memory.shape;np,nt,_=probes.shape
    m=memory.reshape(-1,width);q=probes.reshape(-1,width)
    d=(m.square().sum(1)[:,None]+q.square().sum(1)[None,:]-2*(m@q.T)).clamp_min_(0)
    logdensity=torch.log(torch.exp(-d/(2*sigma*sigma)).reshape(nmem,nfix,np,nt).mean(1)+1e-12).double().sum(-1)
    score,index=logdensity.max(dim=0)
    return score,index,logdensity

def encode_inputs(e,seed,condition,noise):
    from design import plan
    p=plan(e.design,seed)
    memory=torch.stack([e.encode(seed,'study',cat,row,condition[0],10,noise) for cat,row in p['study_order']])
    probes=torch.stack([e.encode(seed,'probe',t['category'],t[which],condition[1],32,noise) for t in p['trials'] for which in ['old','new']])
    return p,memory,probes
