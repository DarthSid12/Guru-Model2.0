import hashlib
import itertools
import random
from common import DESIGNS,SOURCES,POOL_SIZES

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def plan(design,seed):
    groups={};study_order=[];trials=[]
    for category,n in DESIGNS[design].items():
        rng=random.Random(seed_for('selection-v1',design,seed,category))
        selected=rng.sample(range(POOL_SIZES[category]),n['pool'])
        studied=selected[:n['study']];novel=selected[n['study']:]
        assert len(novel)==n['trials']
        tested_old=rng.sample(studied,n['trials']);rng.shuffle(novel)
        groups[category]={'studied':studied,'novel':novel,'tested_old':tested_old}
        study_order.extend((category,i) for i in studied)
        trials.extend({'category':category,'old':a,'new':b} for a,b in zip(tested_old,novel))
    rng=random.Random(seed_for('order-v1',design,seed))
    if len(groups)==1:rng.shuffle(study_order)
    else:
        while True:
            rng.shuffle(study_order)
            if max(len(list(g)) for _,g in itertools.groupby(c for c,i in study_order))<=4:break
    rng.shuffle(trials)
    assert len(study_order)==40 and len(trials)==24
    return {'design':design,'seed':seed,'groups':groups,'study_order':study_order,'trials':trials}
