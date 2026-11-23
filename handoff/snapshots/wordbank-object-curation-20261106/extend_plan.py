"""Replenish all parent photos with nonhuman-animal label flags, preserving budgets."""
import collections,gzip,hashlib,json,shutil
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
OLD=ROOT.parent/'anvil-wordbank-lp-20261020'
def main():
 prior=json.loads((ROOT/'SOURCE_PLAN.json').read_text())
 snapshot=ROOT/'phase1';snapshot.mkdir(exist_ok=True)
 for name in ['SOURCE_PLAN.json','image_candidates.json.gz','FRESH_SOURCES.json']:
  shutil.copyfile(ROOT/name,snapshot/name)
 parent=json.loads((PARENT/'PHOTO_SPLITS.json').read_text())['images'];old=json.loads((OLD/'PHOTO_SPLITS.json').read_text())['images']
 classes={r['class_name'] for r in parent}
 flags=json.loads((ROOT/'ANIMAL_SOURCE_FLAGS.json').read_text())['parent_flags'];flag_counts=collections.Counter(r['class_name'] for r in flags)
 db=duckdb.connect(str(ROOT/'source-index.duckdb'),read_only=True);animal={r[0] for r in db.sql('SELECT DISTINCT ImageID FROM animal_positives').fetchall()}
 banned={r['image_id'] for r in parent+old}|animal
 staged=json.loads((ROOT/'FRESH_SOURCES.json').read_text())['images'];stage_owner={r['image_id']:r['class_name'] for r in staged}
 candidates=json.load(gzip.open(ROOT/'image_candidates.json.gz','rt'))
 old_candidates=json.load(gzip.open(OLD/'wordbank/image_candidates.json.gz','rt'))
 for cl in classes-set(candidates):candidates[cl]=old_candidates[cl]
 for cl,rows in candidates.items():candidates[cl]=[r for r in rows if r['image_id'] not in banned and (r['image_id'] not in stage_owner or stage_owner[r['image_id']]==cl)]
 owners={}
 for cl,rows in candidates.items():
  for r in rows:
   iid=r['image_id'];key=(0 if stage_owner.get(iid)==cl else 1,len(rows),hashlib.sha256(('object-curation-owner-v2|'+iid+'|'+cl).encode()).digest())
   if iid not in owners or key<owners[iid][0]:owners[iid]=(key,cl)
 for cl,rows in candidates.items():candidates[cl]=[r for r in rows if owners[r['image_id']][1]==cl]
 new_classes=set(prior['new_targets']);targets={cl:(800 if cl=='glass' else flag_counts[cl]+(100 if cl in new_classes else 0)) for cl in sorted(classes)}
 assert sum(targets.values())==4051
 for cl,n in targets.items():assert len(candidates[cl])>=n,(cl,n,len(candidates[cl]))
 with gzip.open(ROOT/'image_candidates.json.gz','wt') as f:json.dump(candidates,f)
 plan={**prior,'status':'ANIMAL_SCREENED_REPLACEMENT_AND_RESERVE_PLAN','phase':2,'new_targets':targets,'target_fresh_photos':sum(targets.values()),'animal_flagged_parent_photos':len(flags),'animal_flagged_non_glass_replacements':sum(flag_counts.values())-flag_counts['glass'],'parent_flag_count_by_class':dict(flag_counts),'candidate_counts':{cl:len(rows) for cl,rows in candidates.items()},'candidate_ownership':'Preserve staged ownership, then prioritize scarcer source pools; fixed object-curation-owner-v2 hash tie-break','animal_screen_scope':'Exclude all 110 nonhuman Animal-hierarchy labels from fresh candidate images; flagged parent images are conservatively superseded even when the animal label may describe a toy/food. This is annotation-based screening, not a claim that every pixel is animal-free.'}
 (ROOT/'SOURCE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
 print('Targets',sum(targets.values()),'new photos; scarce pools',[(cl,targets[cl],len(candidates[cl])) for cl in ['blanket','teddybear','noodles','carrots','glass']],flush=True)
if __name__=='__main__':main()
