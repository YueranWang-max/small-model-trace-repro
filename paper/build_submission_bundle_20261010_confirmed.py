"""Whitelist the actual manuscript dependencies; package only compiled source."""
from pathlib import Path
import re,json,hashlib,shutil,zipfile,sys
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'submission_candidate_20261010_confirmed'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect():
 seen=set();figs=set();bib=set()
 def visit(name):
  p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.exists(),name
  if p in seen:return
  seen.add(p);s=p.read_text(encoding='utf-8')
  for x in re.findall(r'\\input\{([^}]+)\}',s):visit(x if x.endswith('.tex') else x+'.tex')
  for x in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):figs.add((ROOT/x).resolve())
  for group in re.findall(r'\\bibliography\{([^}]+)\}',s):
   for x in group.split(','):bib.add((ROOT/(x+'.bib')).resolve())
 visit('main.tex');files=seen|figs|bib
 assert all(p.exists() and p.is_relative_to(ROOT) for p in files)
 return sorted(files),len(seen),len(figs)
def main():
 files,nt,nf=collect();OUT.mkdir(exist_ok=True)
 if '--zip' not in sys.argv:
  for p in files:
   dst=OUT/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  manifest=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in files]
  (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
  print(json.dumps({'tex_files':nt,'figures':nf,'whitelisted_source_files':len(files),'action':'Copied actual dependencies; compile before packaging'},indent=2));return
 for p in files:assert sha(p)==sha(OUT/p.relative_to(ROOT)),str(p)
 bbl=OUT/'main.bbl';assert bbl.exists()
 assert bbl.stat().st_mtime>=max((OUT/p.relative_to(ROOT)).stat().st_mtime for p in files if p.suffix in ['.tex','.bib'])
 entries=[OUT/p.relative_to(ROOT) for p in files]+[bbl]
 dest=ROOT/'arxiv_source_candidate_20261010_confirmed.zip'
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(entries):z.write(p,p.relative_to(OUT).as_posix())
 text=(ROOT/'main.tex').read_text(encoding='utf-8')
 report={'archive':dest.name,'sha256':sha(dest),'entries':[{'path':p.relative_to(OUT).as_posix(),'sha256':sha(p)} for p in sorted(entries)],'author_placeholder': 'Author details pending verification' in text,'local_compile_only':True,'arxiv_server_compile':False,'not_acceptance_certification':True}
 (ROOT/'arxiv_source_candidate_manifest_20261010.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'zip':dest.name,'entries':len(entries),'author_placeholder':report['author_placeholder']},indent=2))
if __name__=='__main__':main()
