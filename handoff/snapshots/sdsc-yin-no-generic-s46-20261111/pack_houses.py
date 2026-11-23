"""Bind the prior verified house pixels/coordinates to this new specification."""
import json
from common import ROOT, sha, atomic
def verify_packed():
 r=json.loads((ROOT/'PACKED_HOUSES.json').read_text())
 assert r['passed'] and r['seed']==20261103 and r['spec_sha256']==sha(ROOT/'SPEC.json')
 assert r['reused_parent_receipt_sha256']==sha(ROOT/'PARENT_PACKED_HOUSES.json')
 for name,digest in r['files'].items():assert sha(ROOT/name)==digest,name
 return r
def main():
 if (ROOT/'PACKED_HOUSES.json').exists():return verify_packed()
 r=json.loads((ROOT/'PARENT_PACKED_HOUSES.json').read_text())
 assert r['passed'] and r['seed']==20261103
 for name,digest in r['files'].items():assert sha(ROOT/name)==digest,name
 r['parent_spec_sha256']=r['spec_sha256'];r['spec_sha256']=sha(ROOT/'SPEC.json')
 r['reused_parent_receipt_sha256']=sha(ROOT/'PARENT_PACKED_HOUSES.json')
 atomic(ROOT/'PACKED_HOUSES.json',r)
 return verify_packed()
if __name__=='__main__':main()
