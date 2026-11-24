"""Deterministic pools, memory selection, and within-category test pairs."""
import hashlib,itertools,random

def seed_for(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def make_plan(name,definition,seed,sizes):
    groups={};study_order=[];trials=[]
    for cat,n in definition.items():
        rng=random.Random(seed_for('selection-v1',name,seed,cat));chosen=rng.sample(range(sizes[cat]),n['pool'])
        studied=chosen[:n['study']];novel=chosen[n['study']:];old=rng.sample(studied,n['trials']);rng.shuffle(novel)
        groups[cat]=dict(studied=studied,novel=novel,tested_old=old);study_order.extend((cat,i) for i in studied)
        trials.extend(dict(category=cat,old=a,new=b) for a,b in zip(old,novel))
    rng=random.Random(seed_for('order-v1',name,seed))
    if len(groups)==1:rng.shuffle(study_order)
    else:
        # Preserve historical RNG ordering where <=4-category runs are feasible.
        lengths=[len(g['studied']) for g in groups.values()]
        possible=max(lengths)<=4*(sum(lengths)-max(lengths)+1)
        if not possible:raise ValueError('Composition cannot interleave memory with at most four consecutive items of one category')
        for _ in range(100000):
            rng.shuffle(study_order)
            if max(len(list(g)) for _,g in itertools.groupby(cat for cat,row in study_order))<=4:break
        else:raise ValueError('Could not interleave this composition within the bounded shuffle budget')
    rng.shuffle(trials);return dict(design=name,seed=seed,groups=groups,study_order=study_order,trials=trials)
