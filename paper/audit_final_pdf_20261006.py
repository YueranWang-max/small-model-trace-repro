"""Inspect compiled PDF text and fonts; does not render or modify any image."""
from pathlib import Path
import json,hashlib,re
import fitz
ROOT=Path(__file__).resolve().parent
p=ROOT/'submission_candidate_20261006/main.pdf';doc=fitz.open(p)
assert p.read_bytes().startswith(b'%PDF-1.7')
text='\n'.join(page.get_text() for page in doc)
assert '??' not in text
normalized=re.sub(r'\s+',' ',text)
assert 'A larger output allowance was tested' in normalized
assert 'A larger configured context enabled' in normalized
fonts={}
for page in doc:
 for entry in page.get_fonts(full=True):
  xref=entry[0]
  if xref not in fonts:
   data=doc.extract_font(xref);fonts[xref]={'name':data[0],'extension':data[1],'type':data[2],'embedded_bytes':len(data[3])}
assert all(v['embedded_bytes']>0 for v in fonts.values())
log=(ROOT/'submission_candidate_20261006/main.log').read_text(encoding='utf-8',errors='replace')
assert not re.search(r'Citation .* undefined|Reference .* undefined|There were undefined references|^! ',log,re.M)
report={'source':'submission_candidate_20261006/main.pdf','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':len(doc),'header':'PDF-1.7','unresolved_reference_markers':False,'embedded_fonts':list(fonts.values()),'all_fonts_embedded':True,'scope':'Local external Tectonic output, no arXiv server compile; PDF image-reading and Fontconfig notices may remain','author_placeholder':'Author details pending verification' in text,'raster_images_by_page':[len(page.get_images(full=True)) for page in doc]}
(ROOT/'revision_evidence_20261006/latest_local_compile.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'PDF {len(doc)} pages, all {len(fonts)} fonts embedded, no undefined citations/references, header 1.7.')
