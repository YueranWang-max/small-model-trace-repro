"""Check archive CRCs, exact membership and SHA-256 without invoking inference."""
from pathlib import Path, PurePosixPath
import hashlib,json,zipfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
reports={}
for filename,manifest,base,extra in [
 ('arxiv_source_candidate_20261006.zip',P/'submission_candidate_20261006/SOURCE_MANIFEST.json',P/'submission_candidate_20261006',{'main.bbl'}),
 ('reproducibility_evidence_candidate_20261006.zip',P/'reproducibility_staging_20261006/EVIDENCE_MANIFEST.json',P/'reproducibility_staging_20261006',{'EVIDENCE_MANIFEST.json'})]:
 entries=json.loads(manifest.read_bytes());expected={x['path']:x for x in entries}
 with zipfile.ZipFile(P/filename) as z:
  names=z.namelist();assert len(names)==len(set(names))
  assert set(names)==set(expected)|extra,(filename,set(names)-set(expected)-extra)
  assert z.testzip() is None
  for name in names:
   q=PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts and '\\' not in name
   data=z.read(name);assert data==(base/name).read_bytes(),name
   if name in expected:
    assert sha(data)==expected[name]['sha256'],name
    if 'bytes' in expected[name]:assert len(data)==expected[name]['bytes']
  reports[filename]={'sha256':sha((P/filename).read_bytes()),'entries':len(names),'bytes':(P/filename).stat().st_size,'crc_passed':True,'exact_manifest_membership':True,'all_member_hashes_passed':True,'archive_bytes_match_staging':True}
 current={(P/'main.tex').name:(P/'main.tex').read_bytes()}
 with zipfile.ZipFile(P/'arxiv_source_candidate_20261006.zip') as z:
  assert b'Author details pending verification' not in z.read('main.tex')
  assert b'Yueran Wang' in z.read('main.tex')
  for row in json.loads((P/'submission_candidate_20261006/SOURCE_MANIFEST.json').read_bytes()):
   assert z.read(row['path'])==(P/row['path']).read_bytes(),row['path']
 # All four current archive copies must match the working originals, including every raw byte.
 study_files=0
 for directory in ['official_graph_confirmatory_20261006','official_graph_paired4b_20261006','output_budget_control_20261006','context_recovery_20261006']:
  original=P.parent/'logs'/directory
  for f in original.rglob('*'):
   if f.is_file():
    rel=f.relative_to(P.parent);assert f.read_bytes()==(P/'reproducibility_staging_20261006'/rel).read_bytes(),str(rel);study_files+=1
reports['current_study_files_checked']=study_files
reports['scope']='CRC, membership and byte/hash consistency of local saved evidence; not external inference authentication or arXiv server approval'
reports['all_passed']=True
(P/'revision_evidence_20261006/evidence_package_checks.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print(json.dumps(reports,indent=2))
