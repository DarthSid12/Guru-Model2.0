"""Exact frozen calibration function reused for a matched sigma comparison."""
def fit_axis(e,s,sigma,condition,target,test_held=None):
    from kde import inputs,score
    results={}
    def accuracy(rate):return sum(x['correct'] for x in results[rate])/(50*12)
    def best(rates):return min(rates,key=lambda p:(abs(accuracy(p)-target),p))
    def evaluate(rates):
        missing=[r for r in rates if r not in results]
        for r in missing:results[r]=[]
        for seed in s['calibration_seeds']:
            e.draws.clear()
            for rate in missing:
                tn=rate if test_held is None else test_held
                p,mem,q=inputs(e,seed,condition,rate,tn)
                indices=[j for i,t in enumerate(p['trials']) if t['category']=='faces' for j in [2*i,2*i+1]]
                values,_=score(q[indices],mem,sigma);ok=values.reshape(12,2)[:,0]>values.reshape(12,2)[:,1]
                results[rate].append(dict(seed=seed,n_trials=12,correct=int(ok.sum())))
    evaluate(s['coarse_noise_grid']);coarse=best(s['coarse_noise_grid'])
    fine=[i/1000 for i in range(max(0,round(coarse*1000)-50),min(500,round(coarse*1000)+50)+1,5)]
    evaluate(fine);fine_best=best(results)
    ultra=[i/1000 for i in range(max(0,round(fine_best*1000)-5),min(500,round(fine_best*1000)+5)+1)]
    evaluate(ultra);selected=best(results)
    curve=[dict(rate=r,accuracy=accuracy(r),seed_counts=results[r]) for r in sorted(results)]
    return dict(condition=condition,target=target,test_noise_held_at=test_held,coarse_best=coarse,fine_grid=fine,fine_best=fine_best,ultrafine_grid=ultra,selected_noise=selected,achieved=accuracy(selected),absolute_target_error=abs(accuracy(selected)-target),target_within_sampled_range=min(x['accuracy'] for x in curve)<=target<=max(x['accuracy'] for x in curve),selected_boundary=selected in [0.,.5],curve=curve)
