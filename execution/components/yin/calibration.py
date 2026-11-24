"""One- or two-parameter fitting with explicit anchor and transfer behavior."""
from .memory import noise_rates

def fit_axis(evaluate,condition,target,test_held,protocol):
    values={}
    def fit(rates):
        for rate in rates:
            if rate not in values:
                p1=rate if test_held is None else test_held;p2=rate
                values[rate]=evaluate(condition,*noise_rates(protocol,condition,p1,p2))
    def best():return min(values,key=lambda p:(abs(values[p]-target),p))
    fit([i/20 for i in range(11)]);b=best()
    fit([i/1000 for i in range(max(0,round(b*1000)-50),min(500,round(b*1000)+50)+1,5)]);b=best()
    fit([i/1000 for i in range(max(0,round(b*1000)-5),min(500,round(b*1000)+5)+1)])
    b=best();return dict(noise=b,condition=condition,target=target,achieved=values[b],curve=values,target_within_sampled_range=min(values.values())<=target<=max(values.values()))
