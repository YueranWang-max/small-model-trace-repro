"""Check reviewed TeX objects against compiled PDF and freeze an explicit content contract.

This bounded audit does not render images, infer scientific validity from compilation,
or assert complete automatic mathematical/semantic equivalence.
"""
from pathlib import Path
import re,json,hashlib,sys,unicodedata
import fitz
P=Path(__file__).resolve().parent
OUT=P/'revision_evidence_20261010';OUT.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
norm=lambda s:re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
files=[];visited=set();events=[]
def arg(s,start):
 assert s[start]=='{'
 depth=1;i=start+1
 while depth:
  if s[i]=='{' and s[i-1]!='\\':depth+=1
  if s[i]=='}' and s[i-1]!='\\':depth-=1
  i+=1
 return s[start+1:i-1],i
def plain(s):
 s=re.sub(r'\\(?:cite\w*|ref|eqref|label|path)(?:\[[^]]*\])*\{[^}]*\}','',s)
 s=s.replace(r'\%', '%').replace(r'\_', '_').replace(r'\,','').replace(r'\!','')
 s=re.sub(r'\\[a-zA-Z]+\*?(?:\[[^]]*\])?','',s)
 return s.replace('{','').replace('}','').replace('$','').replace('~',' ')
def visit(path):
 path=path.resolve();assert path.is_relative_to(P) and path.exists()
 assert path not in visited,'Repeated input would duplicate content: '+str(path)
 visited.add(path);s=path.read_text(encoding='utf-8');files.append((path,s))
 pattern=r'\\input\{([^}]+)\}|\\begin\{(figure|table)\}(.*?)\\end\{\2\}'
 for match in re.finditer(pattern,s,re.S):
  if match[1]:visit(P/(match[1] if match[1].endswith('.tex') else match[1]+'.tex'))
  else:events.append((match[2],path,match[3]))
visit(P/'main.tex')
pdf=P/'submission_candidate_20261010_confirmed/main.pdf';doc=fitz.open(pdf)
assert pdf.read_bytes()==(P/'main.pdf').read_bytes(),'Displayed PDF differs from candidate'
pages=[p.get_text() for p in doc];alltext=norm(' '.join(pages))
aux=(P/'submission_candidate_20261010_confirmed/main.aux').read_text(encoding='utf-8')
auxlabels={k:(n,int(pg)) for k,n,pg in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
sections=[]
for path,s in files:
 for m in re.finditer(r'\\(?:section|subsection)\*?\{',s):
  title,_=arg(s,m.end()-1);assert norm(plain(title)) in alltext,title
  sections.append({'source':str(path.relative_to(P)),'title':title,'present_in_pdf':True})
figures=[];tables=[];unhandled_rows=[];numeric_checks=[]
for kind,path,s in events:
 lab=re.search(r'\\label\{([^}]+)\}',s);assert lab,kind
 label=lab[1];assert label in auxlabels,label
 number,pg=auxlabels[label];page=doc[pg-1];text=norm(page.get_text())
 assert norm(kind+' '+number+':') in text,(label,'Missing visible numbered caption')
 cap=re.search(r'\\caption(?:\[[^]]*\])?\{',s);assert cap,label
 caption,_=arg(s,cap.end()-1)
 anchor=' '.join(plain(caption).split()[:10]);assert norm(anchor) in text,(label,anchor)
 item={'source':str(path.relative_to(P)),'label':label,'number':number,'pdf_page':pg,'caption_sha256':sha(caption.encode()),'caption_anchor_present':True}
 if kind=='figure':
  names=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s);assert len(names)==1,label
  f=P/names[0];src=fitz.open(f);assert len(src)==1
  w,h=src[0].rect.width,src[0].rect.height
  # Each current figure is an imported vector PDF Form, not merely a caption.
  candidates=[x for x in page.get_xobjects() if x[2]==0 and abs(fitz.Rect(x[3]).width-w)<1 and abs(fitz.Rect(x[3]).height-h)<1]
  assert len(candidates)==1,(label,'Missing/ambiguous imported figure form')
  x=candidates[0];assert doc.xref_stream(x[0]),label
  # Visible text from the source graphic must also survive in the caption's page.
  labels=[line for line in src[0].get_text().splitlines() if len(norm(line))>=4]
  missing=[line for line in labels if norm(line) not in text];assert not missing,(label,missing)
  item.update(asset=names[0],asset_sha256=sha(f.read_bytes()),vector_form_xref=x[0],vector_form_bytes=len(doc.xref_stream(x[0])),source_graphic_text_lines_checked=len(labels))
  figures.append(item)
 else:
  tab=re.search(r'\\begin\{tabular\}(.*?)\\end\{tabular\}',s,re.S);assert tab,label
  checked=[]
  for row in tab[1].split(r'\\'):
   if '&' not in row:continue
   row=re.sub(r'\\(?:toprule|midrule|bottomrule)','',row).strip()
   # The first row may carry the column specification; only count data rows below.
   if '\\arraybackslash' in row or row.startswith('{'):continue
   if re.search(r'\\(?:frac|sum|begin|sqrt)',row):unhandled_rows.append({'label':label,'row':row});continue
   cells=[plain(v).strip() for v in row.split('&')]
   # Numeric-cell and row-label presence is a bounded check, not a table-layout proof.
   for v in cells:
    if norm(v):assert norm(v) in text,(label,'Missing table cell',v)
    if re.fullmatch(r'[+-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?',v):
     compact=re.sub(r'\s+','',unicodedata.normalize('NFKD',page.get_text()).replace('−','-'))
     assert v in compact,(label,'Missing exact numeric token',v)
     numeric_checks.append({'table':label,'value':v,'page':pg})
   checked.append(cells)
  item.update(tabular_sha256=sha(tab[1].encode()),rows_checked=len(checked),cells=checked)
  tables.append(item)
assert len(figures)==11
assert len({v['asset_sha256'] for v in figures})==11,'Duplicate source figure asset'
form_total=sum(len([x for x in p.get_xobjects() if x[2]==0]) for p in doc)
assert form_total==11,'Unexpected duplicated/missing top-level vector forms'
math=[];inline_math=[]
for path,s in files:
 for m in re.finditer(r'\\\[(.*?)\\\]|\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\2\}',s,re.S):
  raw=m[0];math.append({'source':str(path.relative_to(P)),'sha256':sha(raw.encode()),'tex':raw})
 for m in re.finditer(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)',s,re.S):
  raw=m[0];inline_math.append({'source':str(path.relative_to(P)),'sha256':sha(raw.encode()),'tex':raw})
joined='\n'.join(s for _,s in files)
keys=sorted({k.strip() for match in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',joined) for k in match.split(',')})
bbl=(P/'submission_candidate_20261010_confirmed/main.bbl').read_text(encoding='utf-8')
bibkeys=re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',bbl)
assert sorted(bibkeys)==keys,(bibkeys,keys)
for i in range(1,len(keys)+1):assert norm(f'[{i}]') in alltext
contract={'basis':'Reviewed current TeX baseline; not an independent gold manuscript or proof of complete semantic equivalence','document_class':re.search(r'\\documentclass(?:\[[^]]*\])?\{([^}]+)\}',joined)[0],'packages':re.findall(r'\\usepackage(?:\[[^]]*\])?\{([^}]+)\}',joined),'source_hashes':[{'path':str(p.relative_to(P)),'sha256':sha(p.read_bytes())} for p,_ in files],'sections':sections,'figures':figures,'tables':tables,'display_math':math,'inline_math':inline_math,'citation_keys':keys,'bibliography_sha256':sha((P/'references.bib').read_bytes())}
protected=OUT/'reviewed_content_contract_confirmed_extension_v3.json'
if '--freeze' in sys.argv:
 assert not protected.exists(),'Do not overwrite the reviewed contract; explicitly version it'
 protected.write_text(json.dumps(contract,indent=2),encoding='utf-8')
else:
 assert protected.exists(),'First freeze a reviewed baseline explicitly'
 assert json.loads(protected.read_bytes())==contract,'Content contract changed; review differences before accepting'
report={'passed':True,'pdf_sha256':sha(pdf.read_bytes()),'pages':len(doc),'section_headings_checked':len(sections),'figures_checked':len(figures),'top_level_vector_forms':form_total,'tables_checked':len(tables),'table_rows_checked':sum(t['rows_checked'] for t in tables),'exact_numeric_tokens_checked':len(numeric_checks),'unhandled_complex_rows':unhandled_rows,'display_math_blocks_protected':len(math),'inline_math_spans_protected':len(inline_math),'cited_keys_and_bibitems_match':keys,'limits':['No Markdown input exists for this manuscript; this is not a Markdown round-trip audit','Formula hashes protect the reviewed TeX, not full rendered-symbol semantic correctness','Table cell presence does not alone establish cell alignment','Counts and hashes do not establish scientific validity or external inference authenticity']}
(OUT/'table_numeric_presence.json').write_text(json.dumps({'scope':'Exact numeric token presence on corresponding table page, preserving decimal/fraction/sign; not full cell-position equivalence','checks':numeric_checks,'passed':True},indent=2),encoding='utf-8')
(OUT/'content_preservation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
