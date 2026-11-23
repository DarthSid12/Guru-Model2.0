"""Screen source images for human-positive nonhuman animal labels, without models."""
import csv,json
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
def main():
 names={i:n for i,n in csv.reader((PARENT/'research/oidv7-class-descriptions.csv').open())}
 ids={n:i for i,n in names.items()};tree=json.loads((PARENT/'research/oid-bbox-hierarchy.json').read_text())
 def descendants(label):
  found=set()
  def walk(node,active=False):
   active=active or node['LabelName']==label
   if active:found.add(node['LabelName'])
   for child in node.get('Subcategory',[]):walk(child,active)
  walk(tree);return found
 labels=descendants(ids['Animal'])
 labels-=descendants(ids.get('Person',''))
 labels-=descendants(ids.get('Toy',''))
 labels.discard(ids.get('Teddy bear',''))
 assert ids['Dog'] in labels and ids['Cat'] in labels and ids.get('Person') not in labels
 print('Nonhuman animal source labels',len(labels),flush=True)
 db=duckdb.connect(str(ROOT/'source-index.duckdb'));db.execute("SET threads=4");db.execute("SET memory_limit='2GB'")
 literal=','.join("'"+i+"'" for i in sorted(labels))
 db.execute(f"CREATE OR REPLACE TABLE animal_positives AS SELECT DISTINCT ImageID,LabelName FROM read_csv(?,header=true,delim=',',quote='',strict_mode=false,parallel=false,columns={{'ImageID':'VARCHAR','Source':'VARCHAR','LabelName':'VARCHAR','Confidence':'DOUBLE'}}) WHERE Confidence=1.0 AND LabelName IN ({literal})",[str(ROOT.parents[1]/'tmp/oidv7-train-annotations-human-imagelabels.csv')])
 parent=json.loads((PARENT/'PHOTO_SPLITS.json').read_text())['images'];lookup={r['image_id']:r for r in parent}
 flagged={}
 for iid,label in db.sql('SELECT * FROM animal_positives').fetchall():
  if iid in lookup:flagged.setdefault(iid,[]).append({'label':label,'name':names.get(label,label)})
 records=[{**lookup[iid],'animal_labels':labels} for iid,labels in flagged.items()]
 report={'source_filter':'Human-positive nonhuman Animal hierarchy, excluding Person and Toy; candidate screen, not a guarantee of animal-free pixels','label_ids':sorted(labels),'label_names':{i:names.get(i,i) for i in sorted(labels)},'parent_flag_count':len(records),'parent_flags':records,'automatic_parent_deletion':False}
 (ROOT/'ANIMAL_SOURCE_FLAGS.json').write_text(json.dumps(report,indent=2)+'\n')
 from collections import Counter
 print('Parent flags',len(records),Counter(r['class_name'] for r in records),flush=True)
if __name__=='__main__':main()
