"""Recompute the original public Wordbank WG comprehension extraction exactly."""
import hashlib,json,shutil
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'anvil-wordbank-lp-20261020'
SOURCE=ROOT/'research/english_american_wg.parquet'
EXPECTED='8fa08d7b716a06e1ddad3ccb4a72ef98235dddf146f605df07c0f2fd7e50a8f5'
def main():
 if not SOURCE.exists():shutil.copyfile(ROOT.parent/'wordbank-curriculum-20261020/research'/SOURCE.name,SOURCE)
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
 db=duckdb.connect();db.read_parquet(str(SOURCE)).create_view('r')
 assert db.sql('SELECT count(*) FROM (SELECT data_id,item_id FROM r GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0
 assert db.sql('SELECT count(*) FROM r WHERE understands IS NULL').fetchone()[0]==0
 columns=['item_id','word','category','age','n_responses','n_administrations','n_understands','proportion']
 stats=[dict(zip(columns,r)) for r in db.sql("SELECT item_id,item_definition,category,age,count(*),count(distinct data_id),sum(CASE WHEN understands THEN 1 ELSE 0 END),avg(CASE WHEN understands THEN 1.0 ELSE 0 END) FROM r WHERE item_kind='word' GROUP BY ALL ORDER BY age,item_definition").fetchall()]
 assert stats==json.loads((BASE/'wordbank/word_proportions.json').read_text())
 (ROOT/'word_proportions.json').write_text(json.dumps(stats,indent=2)+'\n')
 info={'status':'EXACT_REEXTRACTION_VERIFIED','source_sha256':EXPECTED,'rows':len(stats),'administrations':db.sql('SELECT count(distinct data_id) FROM r').fetchone()[0],'words':len({r['word'] for r in stats}),'ages':sorted({r['age'] for r in stats}),'agrees_with_original_extraction':True,'source':'Existing pinned public source; new web download unavailable','orientational_eligibility_filter':False}
 (ROOT/'EXTRACTION_VERIFIED.json').write_text(json.dumps(info,indent=2)+'\n');print(info)
if __name__=='__main__':main()
