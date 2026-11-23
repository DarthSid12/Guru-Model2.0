import functools,gc,hashlib,itertools,json,math,sys,time
from recipe import ROOT,CODE,CATEGORIES,COUNTS,arguments
from common import atomic

def main():
    import torch
    variant,seed=sys.argv[1],int(sys.argv[2]);assert variant in ('lp','cnn')
    sys.path.insert(0,str(CODE));from training import train
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    out=ROOT/'smoke'/f'{variant}_s{seed}';assert not out.exists()
    args=arguments(variant,seed,out)
    def replace(flag,values):
        begin=args.index(flag)+1;end=begin
        while end<len(args) and not args[end].startswith('--'):end+=1
        args[begin:end]=values
    replace('--curriculum-stages',[','.join(f'{c}={n}' for c,n in zip(CATEGORIES,COUNTS[-1]))])
    replace('--curriculum-epochs',['2']);replace('--curriculum-image-caps',['none'])
    replace('--steps-per-epoch',['8']);args.extend(['--max-images-per-class',*[f'{c}=2' for c in CATEGORIES]])
    sys.argv=[args[0],*args[1:]]
    evaluate=train.evaluate;train.evaluate=lambda model,loader,*a,**kw:evaluate(model,itertools.islice(loader,2),*a,**kw)
    initial={};durations=[];timing={};constructor=train.Model;step=torch.optim.AdamW.step
    def pre_forward(model,inputs):
        if model.training:torch.cuda.synchronize();timing['start']=time.perf_counter()
    def factory(*a,**kw):
        model=constructor(*a,**kw);name,param=next(model.named_parameters());initial.update(name=name,value=param.detach().cpu().clone());model.register_forward_pre_hook(pre_forward);return model
    train.Model=factory
    @functools.wraps(step)
    def timed(opt,*a,**kw):
        value=step(opt,*a,**kw);torch.cuda.synchronize();durations.append(time.perf_counter()-timing.pop('start'));return value
    torch.optim.AdamW.step=timed
    save=train.save_resume_state
    class Saved(Exception):pass
    def stop(path,model,optimizer,**state):
        save(path,model,optimizer,**state);assert state['epoch']==1;raise Saved()
    train.save_resume_state=stop
    try:train.main()
    except Saved:pass
    else:raise AssertionError('No checkpoint saved')
    ck=torch.load(out/'checkpoint_last.pth',map_location='cpu',weights_only=False)
    assert ck['epoch']==1 and ck['optimizer']['state'] and ck['rng']
    assert not torch.equal(ck['model'][initial['name']],initial['value'])
    del ck;gc.collect();torch.cuda.empty_cache();train.save_resume_state=save
    train.main();ck=torch.load(out/'checkpoint_last.pth',map_location='cpu',weights_only=False)
    assert [h['epoch'] for h in ck['history']]==[1,2]
    assert len(durations)==16 and all(math.isfinite(t) and t>0 for t in durations)
    assert all(math.isfinite(h['train_loss']) for h in ck['history'])
    atomic(out/'SMOKE_PASSED.json',dict(variant=variant,seed=seed,initial_weight_sha256=hashlib.sha256(initial['value'].numpy().tobytes()).hexdigest(),updates=16,resume_history=[1,2],optimizer_and_rng_preserved=True,weights_changed=True,finite_losses=True,gpu=torch.cuda.get_device_name(0),peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,step_seconds=durations))
    print('OPTIMIZER AND FULL-STATE RESUME SMOKE PASSED',variant,flush=True)

if __name__=='__main__':main()
