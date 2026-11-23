"""Review every parent glass photo before replacing mismatched sources."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
OUT=ROOT.parents[1]/'output/wordbank-object-curation-20261106'
def main():
 path=PARENT/'PHOTO_SPLITS.json';parent=json.loads(path.read_text())
 rows=sorted([r for r in parent['images'] if r['class_name']=='glass'],key=lambda r:r['image_id'])
 assert len(rows)==700
 receipt={'parent_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'images':[{'review_index':i,**r} for i,r in enumerate(rows)]}
 (ROOT/'GLASS_REVIEW_INDEX.json').write_text(json.dumps(receipt,indent=2)+'\n')
 font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',13)
 folder=OUT/'glass-parent-review';folder.mkdir(parents=True,exist_ok=True)
 for start in range(0,700,64):
  im=Image.new('RGB',(8*150,8*174+24),'white');d=ImageDraw.Draw(im);d.text((5,3),f'Parent glass photos {start:03d}–{min(start+63,699):03d}',font=font,fill='black')
  for offset,r in enumerate(rows[start:start+64]):
   x=offset%8*150;y=offset//8*174+24
   with Image.open(r['image_path']) as p:p.load();im.paste(p.resize((140,140)),(x+5,y))
   d.text((x+5,y+142),f"{start+offset:03d}  {r['image_id'][:10]}",font=font,fill='black')
  im.save(folder/f'{start//64+1:02d}.jpg',quality=95)
 print('700 parent glass photos indexed on 11 review sheets',flush=True)
if __name__=='__main__':main()
