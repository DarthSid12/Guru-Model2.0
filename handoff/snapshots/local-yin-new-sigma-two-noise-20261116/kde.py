import torch
from design import plan

def inputs(e,seed,condition,study_noise,test_noise):
    p=plan(e.design,seed)
    memory=torch.stack([e.encode(seed,'study',c,r,condition[0],10,study_noise) for c,r in p['study_order']])
    probes=torch.stack([e.encode(seed,'probe',t['category'],t[w],condition[1],32,test_noise) for t in p['trials'] for w in ['old','new']])
    return p,memory,probes

def score(probes,memory,sigma):
    n,nfix,width=memory.shape;np,nt,_=probes.shape
    m=memory.reshape(-1,width);q=probes.reshape(-1,width)
    d=(m.square().sum(1)[:,None]+q.square().sum(1)[None,:]-2*(m@q.T)).clamp_min_(0)
    density=torch.exp(-d/(2*sigma*sigma)).reshape(n,nfix,np,nt).mean(1)
    logs=torch.log(density+1e-12).double().sum(-1)
    best,index=logs.max(0)
    assert torch.isfinite(best).all()
    return best,index

def noise_assignment(condition,p1,p2):
    return (p1 if condition[0]=='U' else p2),p1
