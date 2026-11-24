"""All ordered target/match/distractor triplets, scalar Pearson reference."""
import hashlib

def seed_for(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def correlation_distance(a,b):
    import numpy as np
    a=a-a.mean();b=b-b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    if denom==0:return 1.0
    return 1.0-float(np.dot(a,b)/denom)

def count_triplets(features,labels):
    import numpy as np
    labels=np.asarray(labels);dist=np.array([[correlation_distance(a,b) for b in features] for a in features]);correct=total=0
    for t in range(len(features)):
        positive=np.flatnonzero((labels==labels[t])&(np.arange(len(features))!=t));negative=np.flatnonzero(labels!=labels[t])
        correct+=int((dist[t,positive][:,None]<dist[t,negative][None,:]).sum());total+=len(positive)*len(negative)
    return correct,total

def evaluate(probabilities,labels,seed,noise):
    import torch
    g=torch.Generator().manual_seed(seed_for('kanwisher-binary-xor-v1',seed));shape=probabilities.shape
    u=torch.rand(shape,generator=g);v=torch.rand(shape,generator=g)
    binary=torch.logical_xor(u<probabilities,v<noise).float()
    return count_triplets(binary.mean(1).numpy(),labels)
