import copy,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from execution.common.cli import ROOT,parse
from execution.components.salience.sampling import sample,validate_sampling
from execution.components.yin.design import make_plan
from execution.components.yin.memory import noise_rates,identity_scores,familiarity
from execution.components.kanwisher.matching import count_triplets,correlation_distance
from execution.components.downloads.sources import extract
from execution.components.model.workflow import command,validate

class Contracts(unittest.TestCase):
    def test_all_cli_defaults_validate_without_inputs(self):
        import importlib
        for name,package in [('download','downloads'),('salience','salience'),('preprocess','preprocessing'),('train','model'),('yin','yin'),('kanwisher','kanwisher')]:
            ap,args,c=parse(name,'test',[]);importlib.import_module(f'execution.components.{package}.workflow').validate(c)
    def test_training_arguments_reach_real_parser(self):
        from training.train import parse_args
        c=json.loads((ROOT/'execution/configs/train.json').read_text());args=command(c)
        with patch('sys.argv',args[1:]):actual=parse_args()
        self.assertEqual(actual.backbone,'resnet18');self.assertEqual(sum(actual.curriculum_epochs),124);self.assertEqual(actual.lr,.001)
        c['epochs']=1
        with self.assertRaises(ValueError):validate(c)
    def test_protocol_assignments(self):
        expected={'single_uu':[(.2,.2)]*4,'study_only_fitted':[(.2,.2),(.2,.2),(.3,.2),(.3,.2)],'two_cross':[(.2,.2),(.2,.3),(.3,.2),(.3,.3)],'two_shared_ui':[(.2,.2),(.2,.2),(.3,.2),(.3,.3)]}
        for protocol,pairs in expected.items():self.assertEqual([noise_rates(protocol,c,.2,.3) for c in ['UU','UI','IU','II']],pairs)
    def test_custom_plan_no_leakage(self):
        definition={'faces':dict(pool=5,study=3,trials=2),'houses':dict(pool=5,study=3,trials=2)}
        a=make_plan('custom',definition,101,{'faces':12,'houses':12});self.assertEqual(a,make_plan('custom',definition,101,{'faces':12,'houses':12}))
        self.assertEqual(len(a['study_order']),6);self.assertEqual(len(a['trials']),4)
        for g in a['groups'].values():self.assertFalse(set(g['studied'])&set(g['novel']));self.assertLessEqual(set(g['tested_old']),set(g['studied']))
    def test_pooling_is_not_maximum(self):
        mem=torch.stack([torch.zeros(1,256),torch.ones(1,256)]);q=torch.stack([torch.zeros(2,256)]);q[0,1]=1
        plan={'groups':{'houses':{}},'study_order':[('houses',0),('houses',1)],'trials':[{'category':'houses'}]}
        separate=identity_scores(q,mem,2).max(0).values;pooled=familiarity(q,mem,2,'pooled_all',plan)
        self.assertGreater(float(pooled[0]),float(separate[0]))
        expected=2*np.log((1+np.exp(-32))/2+1e-12);self.assertAlmostEqual(float(pooled[0]),expected,places=6)
    def test_sampling_region_and_repeatability(self):
        images=torch.rand(2,3,32,32)
        for method in ['uniform','grid']:
            c=dict(method=method,num_points=9,edge_margin=.25,attention_center=None,seed=42,batch_size=2,device='cpu');validate_sampling(c)
            torch.manual_seed(42);a=sample(images,c);torch.manual_seed(42);b=sample(images,c)
            self.assertTrue(torch.equal(a,b));self.assertTrue(bool(((a>=8)&(a<=24)).all()))
    def test_matching_all_ordered_triplets_and_ties(self):
        rng=np.random.default_rng(42);x=rng.random((6,256),dtype=np.float32);labels=np.repeat(np.arange(3),2);ok=0;total=0
        for t in range(6):
            for m in range(6):
                for d in range(6):
                    if t!=m and labels[t]==labels[m] and labels[t]!=labels[d]:ok+=correlation_distance(x[t],x[m])<correlation_distance(x[t],x[d]);total+=1
        self.assertEqual(count_triplets(x,labels),(ok,total));self.assertEqual(total,24)
        self.assertEqual(count_triplets(np.ones_like(x),labels),(0,24))
    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','no')
            with self.assertRaises(ValueError):extract(archive,root/'out')
            self.assertFalse((root/'escape.txt').exists())
if __name__=='__main__':unittest.main()
