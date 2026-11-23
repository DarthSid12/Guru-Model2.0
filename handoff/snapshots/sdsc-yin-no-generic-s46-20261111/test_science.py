import ast,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from common import DESIGNS,POOL_SIZES,PROTOCOLS,CONDITIONS,rates
from design import plan
from engine import Experiment
from run import fit

class Science(unittest.TestCase):
 def test_all_design_counts_and_identity_separation(self):
  for design,counts in DESIGNS.items():
   for seed in range(101,151):
    p=plan(design,seed);self.assertEqual(p,plan(design,seed));self.assertEqual(len(p['study_order']),40);self.assertEqual(len(p['trials']),24)
    self.assertEqual(len(set(p['study_order'])),40)
    for cat,n in counts.items():
     g=p['groups'][cat];self.assertEqual(len(g['studied']),n['study']);self.assertEqual(len(g['novel']),n['trials']);self.assertFalse(set(g['studied'])&set(g['novel']))
     self.assertTrue(set(g['tested_old'])<=set(g['studied']));self.assertEqual(len(set(g['studied']+g['novel'])),n['pool'])
     self.assertTrue(all(0<=x<POOL_SIZES[cat] for x in g['studied']+g['novel']))
     self.assertEqual(sum(t['category']==cat for t in p['trials']),n['trials'])
  used={i for s in range(101,151) for i in plan('houses_only',s)['groups']['houses']['studied']}
  self.assertEqual(used,set(range(POOL_SIZES['houses'])))
 def test_noise_mapping(self):
  expected={'single_uu':[(.2,.2)]*4,'study_only_fitted':[(.2,.2),(.2,.2),(.4,.2),(.4,.2)],'two_cross':[(.2,.2),(.2,.4),(.4,.4),(.4,.2)]}
  for protocol in PROTOCOLS:self.assertEqual([rates(protocol,c,.2,.4) for c in CONDITIONS],expected[protocol])
 def test_original_kernel_agreement(self):
  source=Path(__file__).resolve().parent/'source/yin_tests/simulate_yin1969_bothnoise.py'
  tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_p_f_given_c','compute_familiarity_score')]
  self.assertEqual(len(nodes),2);scope={'torch':torch,'_FAST_FAMILIARITY':True};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
  g=torch.Generator().manual_seed(319);memory=torch.randint(0,2,(40,10,256),generator=g).float();probes=torch.randint(0,2,(5,32,256),generator=g).float()
  bank={str(i):m for i,m in enumerate(memory)};expected=torch.tensor([scope['compute_familiarity_score'](f,bank,2) for f in probes],dtype=torch.float64)
  self.assertTrue(torch.allclose(Experiment.familiarity(probes,memory),expected,rtol=0,atol=1e-6))
 def test_cross_category_bank(self):
  memory=torch.zeros(40,10,256);memory[-1]=1;probe=torch.ones(1,32,256)
  self.assertGreater(Experiment.familiarity(probe,memory).item(),Experiment.familiarity(probe,memory[:-1]).item())
 def test_bernoulli_xor_distribution_and_pairing(self):
  e=Experiment.__new__(Experiment);e.design='faces_houses';e.draws={};e.probs={('faces','U'):torch.full((1,32,256),.2)}
  vals=[e.encode(s,'study','faces',0,'U',32,.25) for s in range(101,201)]
  self.assertAlmostEqual(torch.stack(vals).mean().item(),.35,delta=.005)
  self.assertTrue(torch.equal(vals[0],e.encode(101,'study','faces',0,'U',32,.25)));self.assertFalse(torch.equal(vals[0],vals[1]))
  self.assertFalse(torch.equal(vals[0],e.encode(101,'probe','faces',0,'U',32,.25)))
 def test_fit_phase_assignments_and_unreachable_target(self):
  calls=[]
  def get(c,s,t):calls.append((c,s,t));return {'means':{'faces':1-s-t}}
  p,f=fit(get,'UU',.4);self.assertAlmostEqual(p,.3);self.assertTrue(all(s==t for c,s,t in calls));calls.clear()
  p,f=fit(get,'II',.4,.2);self.assertAlmostEqual(p,.4);self.assertTrue(all(t==.2 for c,s,t in calls));calls.clear()
  p,f=fit(lambda c,s,t:{'means':{'faces':.5}},'UU',.9);self.assertEqual(f['status'],'target_outside_sampled_range')
 def test_pool_shape_guards(self):
  from common import SOURCES
  with tempfile.TemporaryDirectory() as folder:
   for cat in ['faces','houses']:
    for ori in ['U','I']:np.save(Path(folder)/(SOURCES[cat]+'_'+ori+'.npy'),np.full((POOL_SIZES[cat],32,256),.4,dtype=np.float32))
   e=Experiment(folder,'faces_houses');a,detail=e.measure(101,'UU',.2,.2,details=True)
   self.assertEqual(detail['memory_items'],40);self.assertEqual(len(detail['scores']),24);self.assertEqual(set(a),{'faces','houses'})
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
