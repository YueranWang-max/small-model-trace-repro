"""Nonvisual integrity checks, compilation, and packaging of existing evidence."""
from pathlib import Path
import hashlib,json,re,shutil,runpy,sys,os,subprocess,zipfile
import fitz
P=Path(__file__).resolve().parent;R=P/'revision_evidence_20261015_refined'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def scientific(s):
 s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',s,flags=re.S)
 return '\n'.join(x.strip() for x in s.splitlines() if x.strip())
for f in (R/'before').rglob('*.tex'):
 p=P/f.relative_to(R/'before');assert scientific(p.read_text(encoding='utf-8'))==scientific(f.read_text(encoding='utf-8')),p
assert sha(P/'references.bib')==sha(R/'before/references.bib')
inputs=json.loads((R/'INPUTS.json').read_text())
for x in inputs:assert sha(Path(x['source']))==sha(Path(x['copy']))==x['sha256']
pilot=json.loads((P/'figures/visual_pilot_20261015/SOURCE_MANIFEST.json').read_text())
for x in pilot:
 assert sha(Path(x['source']))==sha(P/'figures/visual_pilot_20261015'/x['figure']/x['file'])==x['sha256']
builder=runpy.run_path(str(P/'build_submission_bundle_20261014_origin.py'));files,nt,nf=builder['collect']();assert nf==14
new=[f for f in files if any(x in str(f) for x in ['refined_r_20261015','visual_pilot_20261015'])];assert len(new)==11
for f in files:
 if f.suffix=='.pdf' and f not in new:assert sha(f)==sha(R/'before'/f.relative_to(P))
checks=[]
for f in new:
 d=fitz.open(f);pg=d[0];assert len(d)==1 and not pg.get_images();assert all(d.extract_font(x[0])[3] for x in pg.get_fonts())
 spans=[s for b in pg.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
 for s in spans:
  x0,y0,x1,y1=s['bbox'];assert min(x0,y0)>=-.5 and x1<=pg.rect.width+.5 and y1<=pg.rect.height+.5,(f,s['text'],s['bbox'])
 checks.append({'figure':str(f.relative_to(P)),'sha256':sha(f),'minimum_font_pt':min(s['size'] for s in spans),'vector':True,'fonts_embedded':True})
print('Source/data/vector checks passed',flush=True)
candidate=P/'submission_candidate_20261014_origin'
env=os.environ.copy();env['TEXINPUTS']='D:/project/tangjie/.local-tex-dependencies/xurl_package/xurl/latex//;'
tex=Path('C:/Users/dragoncat/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
commands=[[str(tex/'xelatex.exe'),'--disable-installer','-interaction=nonstopmode','-halt-on-error','main.tex'],[str(tex/'bibtex.exe'),'main'],[str(tex/'xelatex.exe'),'--disable-installer','-interaction=nonstopmode','-halt-on-error','main.tex'],[str(tex/'xelatex.exe'),'--disable-installer','-interaction=nonstopmode','-halt-on-error','main.tex']]
for i,c in enumerate(commands):
 print('Compile pass',i+1,flush=True)
 with (R/f'compile_{i+1}.log').open('w',encoding='utf-8') as out:subprocess.run(c,cwd=candidate,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
doc=fitz.open(candidate/'main.pdf');text=''.join(p.get_text() for p in doc)
assert set(map(int,re.findall(r'Figure\s+(\d+):',text)))==set(range(1,15))
assert not any(p.get_images() for p in doc)
assert all(doc.extract_font(f[0])[3] for p in doc for f in p.get_fonts())
log=(candidate/'main.log').read_text(errors='replace');bad=[x for x in ['Overfull','Underfull','undefined','Float too large'] if x in log];assert not bad,bad
bbl=(candidate/'main.bbl').read_text();assert len(re.findall(r'\\bibitem',bbl))==12 and 'tang2026trace' in bbl.split(r'\bibitem')[1]
assert '0.1161' in text and '39.8%' in text
pages={};font_minima={}
for i,p in enumerate(doc):
 for no in map(int,re.findall(r'Figure\s+(\d+):',p.get_text())):
  pages[no]=i+1
  if no in [1,2,4]:continue
  ss=[s for b in p.get_text('dict')['blocks'] for ln in b.get('lines',[]) for s in ln['spans'] if 'Arial' in s['font']]
  font_minima[no]=round(min(s['size'] for s in ss),3)
assert min(font_minima.values())>=6.0
shutil.copy2(candidate/'main.pdf',P/'main.pdf')
qa={'pages':len(doc),'figure_pages':pages,'revised_figures':11,'retained_figures_byte_identical':3,'data_files_byte_identical':len(inputs)+len(pilot),'scientific_body_unchanged':True,'references':12,'fonts_embedded':True,'raster_images':0,'layout_warnings':bad,'new_model_calls':0,'new_inferential_tests':0,'minimum_placed_fonts_pt':font_minima,'figures':checks,'pdf_sha256':sha(P/'main.pdf'),'arxiv_server_tested':False}
(R/'FINAL_QA.json').write_text(json.dumps(qa,indent=2))
sys.argv=[str(P/'build_submission_bundle_20261014_origin.py'),'--zip'];builder['main']()
shutil.copy2(P/'arxiv_source_candidate_20261014_origin.zip',P/'arxiv_source_candidate_20261015_refined.zip')
deliver=set(files)
for f in files:
 if f.suffix=='.pdf':
  for pattern in ['*.csv','*.R','*.png','*.opju','*verify_report.json','*MANIFEST*.json','CAPTIONS.md']:
   deliver.update(f.parent.glob(pattern))
for name in ['draw_refined_20261015.R','export_refined_20261015.R','render_refined_20261015.ps1','prepare_refined_20261015.py','adopt_refined_20261015.py','finish_refined_20261015.py','FIGURE_CONTRACTS_REFINED_20261015.md','draw_visual_pilot_20261015.R','export_visual_pilot_20261015.R','render_visual_pilot_20261015.ps1','prepare_visual_pilot_20261015.py','draw_method_pipeline_20261014.R','export_method_arrowfix_20261014.R','draw_confirmation_composite_20261014.R','export_confirmation_artifacts_20261014.R']:
 deliver.add(P/name)
deliver.update([R/'FINAL_QA.json',R/'INPUTS.json',R/'CAPTIONS.json',P/'figures/visual_pilot_20261015/SOURCE_MANIFEST.json'])
manifest={f.relative_to(P).as_posix():sha(f) for f in sorted(deliver)}
dest=P/'editable_figures_20261015_refined.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
 for name in manifest:z.write(P/name,name)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
print(json.dumps({k:v for k,v in qa.items() if k!='figures'}),flush=True)
