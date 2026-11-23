"""Frozen Set A target matching; one allocated GPU, paired Monte Carlo seeds."""
import ast
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent
PARENT=ROOT
CODE=ROOT/'source'
EXPECTED_ROOT=Path('/expanse/lustre/projects/sdp159/sagrawal5/kanwisher-setA-resnet18-20261106')
CHECKPOINTS={'lp':'874c52d7264c43ef587d24e0be16783404efd428bad3aba0ee6d361b8e322a72','cnn':'aee175bcd2f972eeab14e938d28377aed9ac1a4a1643b5a7579d10e41a480f6a'}
SEEDS=list(range(101,151))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def prepare():
    """Read-only provenance checks, no model inference on login nodes."""
    assert ROOT==EXPECTED_ROOT and not (ROOT/'SPEC.json').exists()
    source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    assert source['identities']==40 and source['photos_per_identity']==5
    assert Counter(r['identity'] for r in source['files'])=={f'setA_{i:02d}':5 for i in range(40)}
    for r in source['files']:assert sha(ROOT/r['path'])==r['sha256']
    reg_path=ROOT/'parent-training-registration.json';reg=json.loads(reg_path.read_text())
    assert reg['seed']==45 and reg['backbone']=='resnet18'
    source_files={}
    # Freeze/check all Python source imports, not just the top-level script.
    for rel,digest in reg['frozen']['source']['files'].items():
        if rel.endswith('.py'):
            assert sha(CODE/rel)==digest,rel
            source_files[str(CODE/rel)]=digest
    models=[]
    for variant in ['lp','cnn']:
        run=PARENT/'models'/f'resnet18_{variant}_e124_s45'
        complete=json.loads((run/'COMPLETE').read_text());cfg=json.loads((run/'config.json').read_text())
        assert complete['artifacts']['epochs']==124
        assert complete['provenance']['registration_sha256']==sha(reg_path)
        assert complete['artifacts']['checkpoint_sha256']==CHECKPOINTS[variant]==sha(run/'checkpoint_last.pth')
        assert cfg['seed']==45 and cfg['variant']==variant and cfg['backbone']=='resnet18' and not cfg['pretrained']
        for name in ['config.json','label_map.json','COMPLETE']:source_files[str(run/name)]=sha(run/name)
        models.append({'model':f'resnet18_{variant}_e124_s45','variant':variant,'run_dir':str(run),'checkpoint_sha256':CHECKPOINTS[variant],'epoch':124,'training_seed':45})
    frozen={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not any(part in p.parts for part in ['__pycache__','logs'])}
    spec={'source_commit':reg['baseline_commit'],'training_registration_sha256':sha(reg_path),'models':models,'files':frozen,'external_source_files':source_files,
      'source_bank':'Authors OSF Set_A_female_faces_200_stim upright: 40 identities x 5 photos',
      'packing':{'input_size':224,'normalization':'training reference Resize(224), CenterCrop(224), RGB','num_coords':32,'edge_margin':.15,'seed':20261106,'batch_size':16},
      'fixations':16,'simulation_seeds':SEEDS,'calibration_seed':42,'calibration_target':.875,'calibration':'nearest upright accuracy, coarse .05/fine .01 in [0,.5]; one rate fixed across upright/inverted',
      'readout':'Bernoulli per fixation, independent XOR per fixation; arithmetic mean over 16 fixation codes; exact scalar Pearson correlation distance; strict ties count incorrect',
      'conditions':['Upright','Inverted'],'identities':40,'photos_per_identity':5,'triplets_per_condition':156000,'expected_result_rows':200,
      'no_retraining':True,'paired_rng':'kanwisher-binary-xor-v1 + simulation seed, shared across variants and orientations','note':'Original Set A photos freshly staged and hash-verified against authors OSF manifest; no claim of matching unavailable historical packed-store coordinates.'}
    atomic(ROOT/'SPEC.json',spec)
    print('FROZEN',sha(ROOT/'SPEC.json'),flush=True)

def verify():
    s=json.loads((ROOT/'SPEC.json').read_text())
    for name,h in s['files'].items():assert sha(ROOT/name)==h,name
    for name,h in s['external_source_files'].items():assert sha(name)==h,name
    for model in s['models']:assert sha(Path(model['run_dir'])/'checkpoint_last.pth')==model['checkpoint_sha256']
    return s

def reference_correlation():
    tree=ast.parse((CODE/'kanwisher_tests/simulate_kanwisher2023.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='correlation_distance')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_scalar_correlation','exec'),ns)
    return ns['correlation_distance']

def triplets(features,labels):
    distances=np.array([[CORRELATION(a,b) for b in features] for a in features])
    labels=np.asarray(labels);correct=trials=0
    for t in range(len(features)):
        positives=np.flatnonzero((labels==labels[t])&(np.arange(len(features))!=t));negatives=np.flatnonzero(labels!=labels[t])
        correct+=int((distances[t,positives][:,None]<distances[t,negatives][None,:]).sum())
        trials+=len(positives)*len(negatives)
    return correct,trials

def scientific_tests():
    rng=np.random.default_rng(20261106);checks=[]
    for name,features in [('random',rng.random((6,256)).astype(np.float32)),('zero_variance',np.ones((6,256),dtype=np.float32)),('ties',np.tile(rng.random((1,256)).astype(np.float32),(6,1)))]:
        labels=np.repeat(np.arange(3),2);correct=trials=0
        for t in range(6):
            for a in range(6):
                if t==a or labels[t]!=labels[a]:continue
                for b in range(6):
                    if labels[b]==labels[t]:continue
                    correct+=int(CORRELATION(features[t],features[a])<CORRELATION(features[t],features[b]));trials+=1
        assert triplets(features,labels)==(correct,trials) and trials==24
        if name!='random':assert correct==0
        checks.append({'fixture':name,'correct':correct,'trials':trials,'exact_reference_agreement':True})
    return checks

def pack(s):
    out=ROOT/'packed/faces_setA/valid'
    if (ROOT/'PACKED.json').exists():
        receipt=json.loads((ROOT/'PACKED.json').read_text());assert receipt['spec_sha256']==sha(ROOT/'SPEC.json')
        for name,h in receipt['files'].items():assert sha(out/name)==h
        return
    from data_preparation.preprocess_fixations import pack_split
    random.seed(20261106);np.random.seed(20261106);torch.manual_seed(20261106);torch.cuda.manual_seed_all(20261106)
    pack_split('faces_setA',str(ROOT/'raw/faces_setA'),str(ROOT/'packed'),'valid',32,224,'cuda:0',16,0,edge_margin=.15)
    meta=json.loads((out/'meta.json').read_text());images=np.load(out/'images.npy',mmap_mode='r');coords=np.load(out/'coords.npy')
    assert images.shape==(200,224,224,3) and coords.shape==(200,32,2)
    assert len(meta['classes'])==40 and Counter(meta['labels'])=={i:5 for i in range(40)}
    assert all(meta['classes'][label]=='setA_'+stem[:2] for label,stem in zip(meta['labels'],meta['stems']))
    assert (coords>=0).all() and (coords<224).all() and (coords>=224*.15-1).all() and (coords<=224*.85+1).all()
    from data_preparation.preprocess_fixations import RawSplitDataset
    ds=RawSplitDataset(str(ROOT/'raw/faces_setA/valid'),224)
    for i in range(200):
        _,x=ds[i];expected=(x.clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        assert np.array_equal(images[i],expected)
    atomic(ROOT/'PACKED.json',{'spec_sha256':sha(ROOT/'SPEC.json'),'files':{n:sha(out/n) for n in ['images.npy','coords.npy','meta.json']},'rows':200,'identities':40,'all_raw_to_packed_pixels_verified':True,'seed':20261106})

def cache(s):
    from training.model import Model
    from training.datasets import _PackedSplit,_crop_at
    from training.salience_trans import OnTheFlyTransform
    sp=_PackedSplit(str(ROOT/'packed'),'faces_setA','valid')
    for record in s['models']:
        out=ROOT/'cache'/record['variant'];out.mkdir(parents=True,exist_ok=True)
        if (out/'DONE.json').exists():
            d=json.loads((out/'DONE.json').read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json')
            for n,h in d['files'].items():assert sha(out/n)==h
            continue
        run=Path(record['run_dir']);cfg=json.loads((run/'config.json').read_text());labels=json.loads((run/'label_map.json').read_text())
        ck=torch.load(run/'checkpoint_last.pth',map_location='cpu',weights_only=False)
        assert len(ck['history'])==124 and ck['history'][-1]['epoch']==124 and ck['stage_idx']==10
        assert len(labels)==2241
        model=Model(num_classes=len(labels),pretrained=False,T=cfg['temperature'],backbone='resnet18');model.load_state_dict(ck['model'],strict=True);del ck
        model.to('cuda:0').eval();model.stochastic=False;checks=[];files={}
        for orient,phase in [('U','valid'),('I','test')]:
            transform=OnTheFlyTransform(phase,record['variant'],'cuda:0').to('cuda:0');values=[]
            with torch.no_grad():
                for i in range(200):
                    crops=torch.stack([_crop_at(sp.images[i],*sp.coords[i,j],180) for j in range(16)]).to('cuda:0')
                    x=transform(crops);_,h,p=model(x,return_rep=True)
                    assert torch.equal(h,p) and p.shape==(16,256) and torch.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    if i in [0,199]:
                        for n in [1,10]:
                            _,_,direct=model(transform(crops[:n]),return_rep=True)
                            assert torch.equal(transform(crops[:n]),x[:n]) and torch.allclose(direct,p[:n],rtol=1e-4,atol=1e-5)
                            checks.append({'orientation':orient,'row':i,'batch':n,'max_abs_error':(direct-p[:n]).abs().max().item()})
                    values.append(p.cpu().numpy())
            filename=orient+'.npy';np.save(out/filename,np.stack(values));files[filename]=sha(out/filename)
            print('CACHED',record['variant'],orient,flush=True)
        atomic(out/'DONE.json',{'spec_sha256':sha(ROOT/'SPEC.json'),'checkpoint_sha256':record['checkpoint_sha256'],'packing_receipt_sha256':sha(ROOT/'PACKED.json'),'files':files,'batch_parity_checks':checks,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'fp32':True,'tf32':False})
        del model;gc.collect();torch.cuda.empty_cache()

def simulate(record):
    variant=record['variant'];out=ROOT/'results'/variant;out.mkdir(parents=True,exist_ok=True)
    probabilities={o:torch.from_numpy(np.load(ROOT/'cache'/variant/(o+'.npy'))) for o in ['U','I']}
    labels=json.loads((ROOT/'packed/faces_setA/valid/meta.json').read_text())['labels']
    draws={}
    def evaluate(seed,orient,p):
        if seed not in draws:
            g=torch.Generator().manual_seed(seed_for('kanwisher-binary-xor-v1',seed))
            draws[seed]=(torch.rand((200,16,256),generator=g),torch.rand((200,16,256),generator=g))
        bu,xu=draws[seed]
        binary=torch.logical_xor(bu<probabilities[orient],xu<p).float()
        correct,n=triplets(binary.mean(1).numpy(),labels)
        assert n==156000
        return correct,n
    if (out/'calibration.json').exists():
        cal=json.loads((out/'calibration.json').read_text());assert cal['spec_sha256']==sha(ROOT/'SPEC.json');best=cal['noise']
    else:
        curve={}
        for p in [round(i*.05,2) for i in range(11)]:
            correct,n=evaluate(42,'U',p);curve[p]=correct/n
        best=min(curve,key=lambda p:(abs(curve[p]-.875),p))
        for p in sorted({round(best+i*.01,2) for i in range(-4,5) if 0<=round(best+i*.01,2)<=.5}):
            if p not in curve:
                correct,n=evaluate(42,'U',p);curve[p]=correct/n
        best=min(curve,key=lambda p:(abs(curve[p]-.875),p))
        cal={'spec_sha256':sha(ROOT/'SPEC.json'),'target':.875,'calibration_seed':42,'noise':best,'achieved':curve[best],'curve':curve,'status':'within_sampled_range' if min(curve.values())<=.875<=max(curve.values()) else 'target_outside_sampled_range'}
        atomic(out/'calibration.json',cal)
    print('CALIBRATED',variant,best,cal['achieved'],cal['status'],flush=True)
    draws.clear();rows=[]
    for seed in SEEDS:
        path=out/f'seed-{seed}.json'
        if path.exists():
            shard=json.loads(path.read_text());assert shard['spec_sha256']==sha(ROOT/'SPEC.json') and shard['noise']==best
            rows.extend(shard['rows']);continue
        seed_rows=[]
        for orient in ['U','I']:
            correct,n=evaluate(seed,orient,best)
            seed_rows.append({'model':record['model'],'variant':variant,'epoch':124,'training_seed':45,'checkpoint_sha256':record['checkpoint_sha256'],'category':'faces_setA','condition':'Upright' if orient=='U' else 'Inverted','simulation_seed':seed,'noise':best,'n_identities':40,'photos_per_identity':5,'fixations':16,'n_trials':n,'correct_triplets':correct,'accuracy_pct':100*correct/n,'error_pct':100*(n-correct)/n})
        atomic(path,{'spec_sha256':sha(ROOT/'SPEC.json'),'noise':best,'rows':seed_rows});rows.extend(seed_rows);draws.pop(seed)
        if seed%10==0:print('SIMULATED',variant,seed,flush=True)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic(out/'DONE.json',{'spec_sha256':sha(ROOT/'SPEC.json'),'rows':len(rows),'results_sha256':sha(out/'results.csv'),'calibration_sha256':sha(out/'calibration.json')})
    return rows

def aggregate(s,rows):
    assert len(rows)==200
    cells=[];contrasts=[]
    for record in s['models']:
        v=record['variant'];by_condition={}
        for condition in ['Upright','Inverted']:
            chunk=[r for r in rows if r['variant']==v and r['condition']==condition]
            assert len(chunk)==50 and {r['simulation_seed'] for r in chunk}==set(SEEDS)
            assert all(r['checkpoint_sha256']==record['checkpoint_sha256'] and r['n_trials']==156000 for r in chunk)
            acc=[r['accuracy_pct'] for r in chunk];err=[r['error_pct'] for r in chunk]
            by_condition[condition]={r['simulation_seed']:r['accuracy_pct'] for r in chunk}
            cells.append({'variant':v,'condition':condition,'noise':chunk[0]['noise'],'n':50,'accuracy_mean_pct':statistics.mean(acc),'accuracy_sd_pct':statistics.stdev(acc),'error_mean_pct':statistics.mean(err),'error_sd_pct':statistics.stdev(err),'sem_pct':statistics.stdev(acc)/50**.5})
        diffs=[by_condition['Upright'][seed]-by_condition['Inverted'][seed] for seed in SEEDS]
        contrasts.append({'variant':v,'upright_minus_inverted_accuracy_pp':statistics.mean(diffs),'sd_pp':statistics.stdev(diffs)})
    with (ROOT/'results/all_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic(ROOT/'results/SUMMARY.json',{'spec_sha256':sha(ROOT/'SPEC.json'),'cells':cells,'paired_inversion_effects':contrasts,'uncertainty':'sample SD across 50 Monte Carlo seeds, conditional on fixed training seed45 checkpoints'})
    atomic(ROOT/'results/VALIDATED.json',{'passed':True,'spec_sha256':sha(ROOT/'SPEC.json'),'packing_receipt_sha256':sha(ROOT/'PACKED.json'),'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'all_trials_sha256':sha(ROOT/'results/all_trials.csv'),'summary_sha256':sha(ROOT/'results/SUMMARY.json'),'rows':200,'all_cells_n50':True,'triplets_per_condition':156000})

def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();return
    assert os.environ.get('SLURM_JOB_ID'),'All inference and simulations require an allocated compute node.'
    start=time.monotonic();s=verify();sys.path.insert(0,str(CODE))
    global np,torch,CORRELATION
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    CORRELATION=reference_correlation();tests=scientific_tests()
    pack(s);cache(s)
    atomic(ROOT/'SMOKE_PASSED.json',{'spec_sha256':sha(ROOT/'SPEC.json'),'tests':tests,'all_cache_batch_parity_checks_passed':True,'packing_verified':True,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)})
    rows=[]
    for record in s['models']:rows.extend(simulate(record))
    aggregate(s,rows)
    print('COMPLETE',len(rows),'rows',round(time.monotonic()-start,2),'seconds',flush=True)

if __name__=='__main__':main()
