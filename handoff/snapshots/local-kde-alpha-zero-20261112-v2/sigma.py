"""Explicit readout variants; binary distances stay identical to the parent."""
import math
import torch
from design import plan

def inputs(experiment,seed,condition,rate):
    p=plan(experiment.design,seed)
    memory=torch.stack([experiment.encode(seed,'study',c,r,condition[0],10,rate) for c,r in p['study_order']])
    probes=torch.stack([experiment.encode(seed,'probe',t['category'],t[w],condition[1],32,rate) for t in p['trials'] for w in ['old','new']])
    return p,memory,probes

def distances(memory,probes):
    ni,nf,width=memory.shape;np,nt,_=probes.shape
    m=memory.reshape(-1,width);f=probes.reshape(-1,width)
    d=(m.square().sum(1)[:,None]+f.square().sum(1)[None,:]-2*m@f.T).clamp_min_(0)
    return d.reshape(ni,nf,np,nt)

def score(d,sigma,mode):
    assert sigma>0 and mode in ['legacy_floor','stable_unfloored']
    if mode=='legacy_floor':
        k=torch.exp(-d/(2*sigma*sigma)).mean(1)
        logk=torch.log(k+1e-12).double()
        floor_fraction=(k<1e-12).double().mean().item()
    else:
        logk=torch.logsumexp(-d.double()/(2*sigma*sigma),dim=1)-math.log(d.shape[1])
        floor_fraction=(logk<math.log(1e-12)).double().mean().item()
    scores=logk.sum(-1).max(0).values
    assert torch.isfinite(scores).all()
    return scores,floor_fraction

def histogram_stats(hist):
    n=sum(hist);assert n>1
    mean=sum(math.sqrt(i)*v for i,v in enumerate(hist))/n
    variance=sum(v*(math.sqrt(i)-mean)**2 for i,v in enumerate(hist))/(n-1)
    def quantile(q):
        # conventional linear interpolation of the sorted Euclidean distances
        pos=(n-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
        def at(k):
            total=0
            for i,v in enumerate(hist):
                total+=v
                if total>k:return math.sqrt(i)
        return at(lo)+(pos-lo)*(at(hi)-at(lo))
    return dict(n_pairs=n,mean_euclidean=mean,sd_euclidean=math.sqrt(variance),median_euclidean=quantile(.5),p05_euclidean=quantile(.05),p95_euclidean=quantile(.95),mean_hamming=sum(i*v for i,v in enumerate(hist))/n,hamming_histogram=hist)

def reference(experiment,rate):
    p=plan(experiment.design,42)
    binary=torch.stack([experiment.encode(42,'study',c,r,'U',10,rate) for c,r in p['study_order']])
    raw=torch.stack([experiment.probs[c,'U'][r,:10] for c,r in p['study_order']])
    ids=torch.arange(40).repeat_interleave(10)
    cats=torch.tensor([0 if c=='faces' else 1 for c,r in p['study_order']]).repeat_interleave(10)
    a,b=torch.triu_indices(400,400,1)
    flat=binary.reshape(400,256)
    ds=(flat.square().sum(1)[:,None]+flat.square().sum(1)[None,:]-2*flat@flat.T).clamp_min_(0)[a,b].long()
    masks={'all':torch.ones(len(a),dtype=torch.bool),'faces_within_photo':(ids[a]==ids[b])&(cats[a]==0),'houses_within_photo':(ids[a]==ids[b])&(cats[a]==1),'faces_between_photos':(ids[a]!=ids[b])&(cats[a]==0)&(cats[b]==0),'houses_between_photos':(ids[a]!=ids[b])&(cats[a]==1)&(cats[b]==1),'cross_category':cats[a]!=cats[b]}
    stats={name:histogram_stats(torch.bincount(ds[mask],minlength=257).tolist()) for name,mask in masks.items()}
    assert stats['all']['n_pairs']==79800
    assert sum(v['n_pairs'] for k,v in stats.items() if k!='all')==79800
    rawd=torch.pdist(raw.reshape(400,256).double())
    rawstats={name:dict(n_pairs=int(mask.sum()),mean_euclidean=rawd[mask].mean().item(),sd_euclidean=rawd[mask].std().item()) for name,mask in masks.items()}
    dbar=stats['all']['mean_euclidean']
    candidates=[dict(label='fixed_2',sigma=2.,multiplier=None)]+[dict(label='distance_x'+str(v),sigma=v*dbar,multiplier=v) for v in [.125,.25,.5,1.,2.]]
    return dict(reference_seed=42,study_only=True,orientation='U',noise_rate=rate,d_bar=dbar,candidates=candidates,binary_distance_stats=stats,raw_sigmoid_distance_stats=rawstats,study_order=p['study_order'])
