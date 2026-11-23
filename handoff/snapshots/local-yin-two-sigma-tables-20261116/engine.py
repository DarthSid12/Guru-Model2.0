from pathlib import Path
import numpy as np
import torch
from common import DESIGNS,SOURCES,POOL_SIZES
from design import plan,seed_for

class Experiment:
    def __init__(self,cache,design):
        self.design=design;self.probs={};self.draws={}
        for category in DESIGNS[design]:
            for orient in ['U','I']:
                a=np.load(Path(cache)/(SOURCES[category]+'_'+orient+'.npy'))
                assert a.shape==(POOL_SIZES[category],32,256) and a.dtype==np.float32
                self.probs[category,orient]=torch.from_numpy(a)

    def encode(self,seed,phase,category,row,orientation,n,p):
        key=(seed,phase,category,row,n)
        if key not in self.draws:
            generator=torch.Generator().manual_seed(seed_for('binary-and-xor-v1',self.design,*key))
            shape=self.probs[category,orientation][row,:n].shape
            self.draws[key]=torch.rand(shape,generator=generator),torch.rand(shape,generator=generator)
        bernoulli_u,xor_u=self.draws[key]
        return torch.logical_xor(bernoulli_u<self.probs[category,orientation][row,:n],xor_u<p).float()

    @staticmethod
    def familiarity(probes,memory,sigma=2):
        count,study_fixations,width=memory.shape
        nprobe,test_fixations,_=probes.shape
        m=memory.reshape(-1,width);f=probes.reshape(-1,width)
        distances=(m.square().sum(1)[:,None]+f.square().sum(1)[None,:]-2*(m@f.T)).clamp_min_(0)
        kernels=torch.exp(-distances/(2*sigma*sigma)).reshape(count,study_fixations,nprobe,test_fixations).mean(1)
        return torch.log(kernels+1e-12).double().sum(-1).max(0).values

    def measure(self,seed,condition,study_noise,test_noise,details=False):
        p=plan(self.design,seed)
        memory=torch.stack([self.encode(seed,'study',c,r,condition[0],10,study_noise) for c,r in p['study_order']])
        probes=[]
        for t in p['trials']:
            probes.extend(self.encode(seed,'probe',t['category'],t[w],condition[1],32,test_noise) for w in ['old','new'])
        scores=self.familiarity(torch.stack(probes),memory).reshape(24,2)
        correct=(scores[:,0]>scores[:,1]).tolist()
        result={c:sum(ok for t,ok in zip(p['trials'],correct) if t['category']==c)/n['trials'] for c,n in DESIGNS[self.design].items()}
        return (result,{'memory_items':len(memory),'scores':scores.tolist()}) if details else result
