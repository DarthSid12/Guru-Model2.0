"""Index human-positive glass labels and their usable human object boxes."""
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
labels="'/m/09tvcd','/m/0h8lw51'"
def main():
 db=duckdb.connect(str(ROOT/'source-index.duckdb'));db.execute('SET threads=4');db.execute("SET memory_limit='2GB'")
 db.execute(f"CREATE OR REPLACE TABLE glass_positives AS SELECT DISTINCT ImageID,LabelName FROM read_csv(?,header=true,delim=',',quote='',strict_mode=false,parallel=false,columns={{'ImageID':'VARCHAR','Source':'VARCHAR','LabelName':'VARCHAR','Confidence':'DOUBLE'}}) WHERE Confidence=1.0 AND LabelName IN ({labels})",[str(ROOT.parents[1]/'tmp/oidv7-train-annotations-human-imagelabels.csv')])
 db.execute(f"CREATE OR REPLACE TABLE glass_boxes AS SELECT ImageID,LabelName,arg_max(struct_pack(xmin:=XMin,xmax:=XMax,ymin:=YMin,ymax:=YMax),(XMax-XMin)*(YMax-YMin)) AS box FROM read_csv_auto(?,header=true) WHERE LabelName IN ({labels}) AND IsGroupOf=0 AND IsDepiction=0 AND IsInside=0 AND IsTruncated=0 AND XMax>XMin AND YMax>YMin AND (XMax-XMin)*(YMax-YMin)>=0.04 GROUP BY ImageID,LabelName",[str(ROOT.parents[1]/'tmp/oidv6-train-annotations-bbox.csv')])
 print(db.sql('SELECT LabelName,count(*) FROM glass_positives GROUP BY 1').fetchall())
if __name__=='__main__':main()
