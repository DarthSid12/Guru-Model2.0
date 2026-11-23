import gc,json,os,sys,time
from pathlib import Path
import numpy as np
from common import ROOT,sha,atomic,spec,SOURCES,POOL_SIZES
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def main():
    s=spec()
    from pack_houses import verify_packed
    verify_packed()
    import torch
    sys.path.insert(0,str(ROOT/'source'))
    from training.model import Model
    from training.datasets import _PackedSplit,_crop_at
    from training.salience_trans import OnTheFlyTransform
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    for record in s['models']:
        out=ROOT/'cache'/record['model'];out.mkdir(parents=True,exist_ok=True)
        if (out/'DONE.json').exists():
            d=json.loads((out/'DONE.json').read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json') and d['model']==record
            for n,h in d['files'].items():assert sha(out/n)==h
            continue
        start=time.monotonic();folder=ROOT/record['run_dir'];cfg=json.loads((folder/'config.json').read_text());labels=json.loads((folder/'label_map.json').read_text())
        ck=torch.load(folder/'checkpoint_last.pth',map_location='cpu',weights_only=False)
        assert cfg['backbone']==record['backbone']=='resnet18' and not cfg.get('pretrained',False)
        assert len(ck['history'])==124 and ck['history'][-1]['epoch']==124 and ck['stage_idx']==10
        assert all(k in ck for k in ['model','optimizer','rng']) and cfg['variant']==record['variant'] and cfg['seed']==record['training_seed']
        model=Model(size=180,num_classes=len(labels),pretrained=False,T=cfg.get('temperature',2),backbone=cfg['backbone']);model.load_state_dict(ck['model'],strict=True);del ck
        model.to('cuda:0').eval();model.stochastic=False;files={};checks=[]
        for category,packed in s['data_roots'].items():
            sp=_PackedSplit(packed,category,'valid');total=POOL_SIZES[next(k for k,v in SOURCES.items() if v==category)];assert len(sp.labels)==total and sp.num_coords>=32
            for orientation,phase in [('U','valid'),('I','test')]:
                tf=OnTheFlyTransform(phase,record['variant'],'cuda:0').to('cuda:0');values=[]
                with torch.no_grad():
                    for row in range(total):
                        crops=torch.stack([_crop_at(sp.images[row],*sp.coords[row,j],180) for j in range(32)]).to('cuda:0');x=tf(crops);_,h,p=model(x,return_rep=True)
                        assert torch.equal(h,p) and p.shape==(32,256) and torch.isfinite(p).all()
                        if row in [0,total-1]:
                            for n in [1,10]:
                                _,_,direct=model(tf(crops[:n]),return_rep=True)
                                assert torch.equal(tf(crops[:n]),x[:n]) and torch.allclose(direct,p[:n],rtol=1e-4,atol=1e-5)
                                checks.append({'category':category,'orientation':orientation,'row':row,'batch':n,'max_abs_error':(direct-p[:n]).abs().max().item()})
                        values.append(p.cpu().numpy())
                name=category+'_'+orientation+'.npy';np.save(out/name,np.stack(values));files[name]=sha(out/name)
                print('CACHED',record['model'],category,orientation,flush=True)
        atomic(out/'DONE.json',{'model':record,'house_packing_sha256':sha(ROOT/'PACKED_HOUSES.json'),'spec_sha256':sha(ROOT/'SPEC.json'),'files':files,'checks':checks,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'deterministic_fp32':True,'tf32':False,'elapsed_seconds':time.monotonic()-start})
        del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
