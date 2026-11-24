"""Encoding and familiarity. No classifier decisions or identity oracle."""
from .design import seed_for

def noise_rates(protocol,condition,p1,p2=None):
    if protocol=='single_uu':return p1,p1
    if protocol=='study_only_fitted':return (p1 if condition[0]=='U' else p2),p1
    if protocol=='two_cross':return (p1 if condition[0]=='U' else p2),(p1 if condition[1]=='U' else p2)
    if protocol=='two_shared_ui':
        if condition=='UI':return p1,p1
        return (p1 if condition[0]=='U' else p2),(p1 if condition[1]=='U' else p2)
    raise ValueError(f'Unknown protocol {protocol}')

class Memory:
    def __init__(self,probabilities,design):self.probabilities=probabilities;self.design=design;self.draws={}
    def encode(self,seed,phase,category,row,orientation,count,noise):
        import torch
        key=(seed,phase,category,row,count)
        if key not in self.draws:
            g=torch.Generator().manual_seed(seed_for('binary-and-xor-v1',self.design,*key));shape=self.probabilities[category,orientation][row,:count].shape
            self.draws[key]=(torch.rand(shape,generator=g),torch.rand(shape,generator=g))
        a,b=self.draws[key]
        return torch.logical_xor(a<self.probabilities[category,orientation][row,:count],b<noise).float()
    def inputs(self,plan,condition,study_noise,test_noise,study_fixations,test_fixations):
        import torch
        seed=plan['seed'];m=torch.stack([self.encode(seed,'study',cat,row,condition[0],study_fixations,study_noise) for cat,row in plan['study_order']])
        q=torch.stack([self.encode(seed,'probe',t['category'],t[k],condition[1],test_fixations,test_noise) for t in plan['trials'] for k in ['old','new']]);return m,q

def identity_scores(probes,memory,sigma):
    import torch
    n,k,width=memory.shape;count,fix,_=probes.shape;m=memory.reshape(-1,width);q=probes.reshape(-1,width)
    d=(m.square().sum(1)[:,None]+q.square().sum(1)[None,:]-2*(m@q.T)).clamp_min_(0)
    density=torch.exp(-d/(2*sigma*sigma)).reshape(n,k,count,fix).mean(1)
    return torch.log(density+1e-12).double().sum(-1)

def familiarity(probes,memory,sigma,readout,plan):
    import torch
    if readout=='identity_max':return identity_scores(probes,memory,sigma).max(0).values
    if readout=='pooled_all':return identity_scores(probes,memory.reshape(1,-1,256),sigma).squeeze(0)
    out=torch.empty(len(probes),dtype=torch.float64)
    for cat in plan['groups']:
        mi=[i for i,(c,row) in enumerate(plan['study_order']) if c==cat];qi=[j for i,t in enumerate(plan['trials']) if t['category']==cat for j in [2*i,2*i+1]]
        m=memory[mi]
        if readout=='pooled_category':m=m.reshape(1,-1,256)
        out[qi]=identity_scores(probes[qi],m,sigma).max(0).values
    return out
