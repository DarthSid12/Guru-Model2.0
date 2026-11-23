"""Re-extract/audit non-animal object curriculum without modifying prior data."""
import csv, hashlib, json, re, sys
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'anvil-wordbank-lp-20261020'
sys.path.insert(0,str(ROOT.parent/'wordbank-curriculum-20261020'))
from build_draft import EXCLUDE,ALIASES,CATEGORIES
AGES=(8,10,12,15,18);COUNTS=(4,8,16,32,64)
NEW_ALIASES={'bottle':'Bottle','beads':'Bead','boots':'Boot','block':'Toy block','keys':'Key','purse':'Handbag','vacuum':'Vacuum cleaner','stroller':'Baby carriage','noodles':'Noodle'}
# Exact/sense-preserving synonyms; source availability remains a separate gate.
# Baby carriage (broad umbrella) needs visual stroller-sense review before use.
EXTRA_EXCLUSIONS={
 'rocking chair':'Subtype of retained basic chair concept',
 'juice':'Beverage contents not reliably distinguished from other drinks/containers',
 'jacket':'Overlapping coat category; preserve original canonical category',
 'paper':'Broad material/component with context-dominated source photos',
 'stroller':'Baby carriage label broader than stroller; defer semantic review',
 'button':'Photo review: ambiguous button label includes interface screenshots and tiny clothing buttons in portraits; exclude this unreliable image mapping',
}

def slug(w):return re.sub('[^a-z0-9]+','_',w.lower()).strip('_')
def main():
 stats=json.loads((ROOT/'word_proportions.json').read_text())
 names={}
 for label,name in csv.reader((ROOT/'research/oidv7-class-descriptions.csv').open()):names.setdefault(name.lower(),[]).append([label,name])
 old=json.loads((BASE/'wordbank/ELIGIBILITY_AUDIT.json').read_text())
 db=duckdb.connect(str(BASE/'wordbank/source-index.duckdb'),read_only=True)
 counts=dict(db.sql('SELECT LabelName,count(*) FROM positives GROUP BY 1').fetchall())
 extra=duckdb.connect(str(ROOT/'source-index.duckdb'),read_only=True)
 counts.update(dict(extra.sql('SELECT LabelName,count(*) FROM extra_positives GROUP BY 1').fetchall()))
 ancestors={}
 def walk(node,parents=()):
  label=node['LabelName'];ancestors.setdefault(label,set()).update(parents)
  for child in node.get('Subcategory',[]):walk(child,parents+(label,))
 walk(json.loads((ROOT/'research/oid-bbox-hierarchy.json').read_text()))
 audit={}
 for r in stats:
  w=r['word']
  if w in audit:continue
  reason=EXTRA_EXCLUSIONS.get(w,EXCLUDE.get(w))
  semantic='inanimate_physical_object'
  if r['category']=='animals' and w!='teddybear':reason='Animal concept excluded by user';semantic='animal'
  elif r['category'] not in CATEGORIES:reason='Outside concrete-object branch (people/body/actions/scenes/etc.)';semantic='outside_branch'
  name={**ALIASES,**NEW_ALIASES}.get(w,w.replace(' (object)','').replace(' (animal)','').replace(' (food)',''))
  hits=names.get(name.lower(),[])
  if len(hits)!=1 and not reason:reason='No unique sense-preserving Open Images label mapping'
  count=counts.get(hits[0][0],0) if len(hits)==1 else 0
  audit[w]={'word':w,'wordbank_category':r['category'],'semantic_type':semantic,'class_name':slug(w),'mapping':hits,'human_positive_source_photos':count,'semantic_exclusion_reason':reason,'toy_category_override':w=='teddybear','orientation_filter':False}
 def choose(min_source):
  selected=[];stages=[];reasons={}
  for age,size in zip(AGES,COUNTS):
   additions=[]
   for r in sorted((r for r in stats if r['age']==age),key=lambda x:(-x['proportion'],x['word'])):
    w=r['word'];a=audit[w]
    if any(s['word']==w for s in selected):continue
    reason=a['semantic_exclusion_reason']
    required_source=900 if w=='noodles' and min_source==1000 else min_source
    if not reason and a['human_positive_source_photos']<required_source:reason=f'Fewer than {required_source} human-positive source photos'
    if not reason:
     label=a['mapping'][0][0]
     collision=next((s for s in selected if label==s['open_images_id'] or label in ancestors.get(s['open_images_id'],set()) or s['open_images_id'] in ancestors.get(label,set())),None)
     if collision:reason='Same/nested category as retained '+collision['word']
    if reason:reasons[w]=reason;continue
    row={**r,'class_name':a['class_name'],'open_images_id':label,'open_images_name':a['mapping'][0][1],'human_positive_source_photos':a['human_positive_source_photos'],'introduced_age_months':age,'semantic_type':a['semantic_type'],'toy_category_override':a['toy_category_override']}
    selected.append(row);additions.append(row)
    if len(selected)==size:break
   stages.append({'age_months':age,'requested_classes':size,'actual_classes':len(selected),'new_words':[r['word'] for r in additions],'all_classes':[r['class_name'] for r in selected]})
  assert all(r['semantic_type']=='inanimate_physical_object' for r in selected)
  assert len({r['open_images_id'] for r in selected})==len(selected)
  return {'preferred_minimum_human_positive_source_photos':min_source,'classes':selected,'stages':stages,'exclusions':reasons,'reviewed_source_reserve_exception':{'word':'noodles','actual_human_positive_photos':919,'minimum_source_gate':900,'reason':'Replace semantically unreliable button mapping while preserving 64 classes and 700 photos/class; exception affects reserve only, not final image budget; decoding and visual review remain mandatory'},'target_classes':64,'actual_classes':len(selected),'source_count_gate_is_not_decoded_photo_readiness':True}
 primary=choose(1000); sensitivity=choose(700)
 oldclasses=json.loads((BASE/'wordbank/CURRICULUM.json').read_text())['classes']
 oldwords={r['word'] for r in oldclasses};newwords={r['word'] for r in primary['classes']}
 record={'status':'WORDS_AND_SOURCE_COUNTS_AUDITED_IMAGES_NOT_READY','language':'English (American)','form':'WG','measure':'understands','source_sha256':'8fa08d7b716a06e1ddad3ccb4a72ef98235dddf146f605df07c0f2fd7e50a8f5','orientation_filter':False,'animals_excluded':True,'teddybear_is_inanimate_toy_not_live_animal':True,'foods_furniture_clothing_vehicles_retained_as_non_animal_physical_objects':True,'ranking':'Retain prior stage words; age-specific comprehension descending, alphabetic tie-break; reject same/ancestor/descendant labels','ages_months':AGES,'requested_stage_counts':COUNTS,'photo_budget_per_class':{'train':500,'valid':100,'test':100},'primary_original_1000_source_gate':primary,'sensitivity_700_source_gate_not_approved_for_production':sensitivity,'removed_old_animal_words':[r['word'] for r in oldclasses if r['category']=='animals' and r['word']!='teddybear'],'retained_old_words':sorted(oldwords&newwords),'new_words':sorted(newwords-oldwords),'new_mapping_aliases':NEW_ALIASES,'visual_review_and_duplicate_screening_required':True}
 for name,obj in [('ELIGIBILITY_AUDIT.json',audit),('CURRICULUM.json',record)]: (ROOT/name).write_text(json.dumps(obj,indent=2)+'\n')
 rows=primary['classes']
 with (ROOT/'OBJECT_WORDS.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (ROOT/'class_order.txt').write_text(''.join(r['class_name']+'\n' for r in rows))
 print(json.dumps({k:record[k] for k in ['removed_old_animal_words','new_words','retained_old_words']},indent=2))
 print('1000-source gate:',primary['actual_classes'],'700-source gate:',sensitivity['actual_classes'])
 print('700-source extra:',sorted({r['word'] for r in sensitivity['classes']}-{r['word'] for r in rows}))
 print('1000 stages',[(s['age_months'],s['actual_classes'],s['new_words']) for s in primary['stages']])
if __name__=='__main__':main()
