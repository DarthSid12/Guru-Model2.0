"""Index human-positive photos and usable target boxes for reviewed object aliases."""
import json
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
LABELS=['/m/015m2','/m/01b638','/m/0by9lp','/m/080hkjn','/m/03v3yw','/m/0d31p','/m/03ylns','/m/012ndj','/m/0mfnf']
def main():
 annotations=ROOT.parents[1]/'tmp'
 db=duckdb.connect(str(ROOT/'source-index.duckdb'));db.execute("SET threads=4");db.execute("SET memory_limit='2GB'")
 labels=','.join("'"+s+"'" for s in LABELS)
 db.execute(f"CREATE OR REPLACE TABLE extra_positives AS SELECT DISTINCT ImageID,LabelName FROM read_csv(?,header=true,delim=',',quote='',strict_mode=false,parallel=false,columns={{'ImageID':'VARCHAR','Source':'VARCHAR','LabelName':'VARCHAR','Confidence':'DOUBLE'}}) WHERE Confidence=1.0 AND LabelName IN ({labels})",[str(annotations/'oidv7-train-annotations-human-imagelabels.csv')])
 db.execute(f"CREATE OR REPLACE TABLE extra_boxes AS SELECT ImageID,LabelName,arg_max(struct_pack(xmin:=XMin,xmax:=XMax,ymin:=YMin,ymax:=YMax),(XMax-XMin)*(YMax-YMin)) AS box FROM read_csv_auto(?,header=true) WHERE LabelName IN ({labels}) AND IsGroupOf=0 AND IsDepiction=0 AND IsInside=0 AND IsTruncated=0 AND XMax>XMin AND YMax>YMin AND (XMax-XMin)*(YMax-YMin)>=0.04 GROUP BY ImageID,LabelName",[str(annotations/'oidv6-train-annotations-bbox.csv')])
 print('Human-positive source counts',db.sql('SELECT LabelName,count(*) FROM extra_positives GROUP BY 1 ORDER BY 1').fetchall())
if __name__=='__main__':main()
