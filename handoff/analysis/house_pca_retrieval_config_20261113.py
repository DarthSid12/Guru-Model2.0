import hashlib,json
from pathlib import Path
B=Path('/Users/darthsid/Documents/ChatGPT/GURU')
P=B/'operations/local-kde-alpha-zero-20261112-v2'
O=B/'output/house-pca-retrieval-20261113'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
 s=json.loads((P/'SPEC.json').read_text())
 for m in s['models']:
  cache=Path(m['cache']);done=json.loads((cache/'DONE.json').read_text());assert done['model']==m['record']
  for n,h in done['files'].items():
   if n.startswith(('faces_cfdWM64','houses_dal64')):assert sha(cache/n)==h
 return s
