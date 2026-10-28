"""Complete, reversible graph relationship indexing crossed with reasoning mode."""
import json,re,time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import run_official_graph_reasoning_mode as client
from run_official_graph_native_probe import ROOT,DATA,OPENER,sha,save,score,oracle
from run_official_graph_identity_recode import recode
from run_official_graph_five_trace import FORMAT
OUT=ROOT/'logs/official_graph_indexed_source';PROTOCOL=ROOT/'protocol_official_graph_indexed_source_20261003.md'
MARKER='Here is the graph to operate on:'
GLOSSARY='The actual graph below may use an edge list or a complete indexed list. u -> v denotes an edge from u to v. OUT u: [v1, v2] denotes edges from u to every listed node. IN v: [u1, u2] denotes edges from every listed node to v. Repeated nodes encode repeated edges. An empty list means no edges in that direction. Apply the requested operation to the actual graph, not the example.'
EDGE=re.compile(r'^([0-9A-Za-z]+) -> ([0-9A-Za-z]+)\s*$')
ARMS=['original_on','original_off','compact_on','compact_off','indexed_on','indexed_off']
def actual_edges(prompt):
    h,_=prompt.rsplit('Operation:',1);body=h.split(MARKER,1)[1]
    return [tuple(m.groups()) for line in body.splitlines() if (m:=EDGE.fullmatch(line.strip()))]
def indexed_prompt(prompt,typ):
    h,q=prompt.rsplit('Operation:',1);prefix,body=h.split(MARKER,1);lines=body.splitlines();positions=[i for i,s in enumerate(lines) if EDGE.fullmatch(s.strip())];assert positions
    lo,hi=positions[0],positions[-1];assert all(not s.strip() or EDGE.fullmatch(s.strip()) for s in lines[lo:hi+1]);edges=actual_edges(prompt);nodes=sorted({v for e in edges for v in e});kind='IN' if typ=='parents' else 'OUT';records=[]
    for node in nodes:
        neighbours=[u for u,v in edges if v==node] if kind=='IN' else [v for u,v in edges if u==node]
        records.append(kind+' '+node+': ['+', '.join(neighbours)+']')
    text=prefix+MARKER+'\n'.join(lines[:lo]+records+lines[hi+1:])+'Operation:'+q
    assert Counter(parse_index(text))==Counter(edges)
    return text
def parse_index(prompt):
    h,_=prompt.rsplit('Operation:',1);body=h.split(MARKER,1)[1];edges=[];nodes=set()
    for line in body.splitlines():
        m=re.fullmatch(r'(IN|OUT) ([0-9A-Za-z]+): \[([^\[\]]*)\]',line.strip())
        if not m:continue
        nodes.add(m[2])
        for v in [x.strip() for x in m[3].split(',') if x.strip()]:edges.append((v,m[2]) if m[1]=='IN' else (m[2],v))
    assert nodes=={n for e in edges for n in e}
    return edges
def restore_score(raw,gold,inverse,error):
    parsed=score(raw,[],error);pred=parsed['prediction'];mapped=[inverse[n] if n in inverse else '<unknown-node:'+n+'>' for n in pred];a,b=set(mapped),set(gold);valid=parsed['valid'];f=2*len(a&b)/(len(a)+len(b)) if a or b else 1.
    return {'prediction':pred,'restored_prediction':mapped,'unknown_prediction_nodes':[n for n in pred if n not in inverse],'valid':valid,'em':float(valid and a==b),'f1':f if valid else 0.}
def main():
    OUT.mkdir(exist_ok=True,parents=True);assert not (OUT/'frozen.json').exists() and not (OUT/'results.json').exists() and not list(OUT.glob('attempt_*.json'))
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_reasoning_mode.py','run_official_graph_identity_recode.py','run_official_graph_native_probe.py','run_official_graph_five_trace.py','run_official_graph_sampling_control.py','data/qwen3_tokenizer_official/tokenizer.json','logs/official_graph_tokenizer/ollama_show.json']}
    tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace','official_graph_identity_recode','official_graph_reasoning_mode']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(p['prompt'].rsplit('Operation:',1)[0].encode()))
    data=pq.read_table(DATA).to_pylist();ps=[]
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(data):
                digest=sha(p['prompt'].rsplit('Operation:',1)[0].encode())
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or digest in used:continue
                used.add(digest);t=time.perf_counter();x=(p['prompt']+'\n'+FORMAT).replace(MARKER,GLOSSARY+'\n'+MARKER,1);coded=recode(x,i,tok);compact=coded['prompts']['compact'];indexed=indexed_prompt(compact,typ);original=actual_edges(x)
                assert Counter((coded['inverse']['compact'][u],coded['inverse']['compact'][v]) for u,v in parse_index(indexed))==Counter(original)
                assert set(oracle(p))==set(p['answer_nodes']);ps.append({'row_index':i,'length_bin':[lo,hi],'graph_head_sha256':digest,**p,'maps':{k:coded['maps'][k] for k in ['original','compact']},'inverse':{k:coded['inverse'][k] for k in ['original','compact']},'prompts':{'original':x,'compact':compact,'indexed':indexed},'index_kind':'IN' if typ=='parents' else 'OUT','compact_dictionary_json_bytes':coded['compact_dictionary_json_bytes'],'preprocess_wall_sec':time.perf_counter()-t});count+=1
                if count==2:break
            assert count==2
    ps=[ps[j] for j in [0,2,4,6,1,3,5,7]]
    with OPENER.open('http://127.0.0.1:11434/api/tags') as response:inventory=json.load(response)
    identity=next(m for m in inventory['models'] if m['name']==client.MODEL);assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
    fr={'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':ps,'identity':identity,'model_inventory':inventory,'seeds':[202610080,202610081],'orders':[[ARMS[(2*i+j)%6:]+ARMS[:(2*i+j)%6] for j in range(2)] for i in range(8)],'ctx':32768,'cap':8192,'regime':client.REGIMES['recommended'],'system':client.SYSTEM,'format_suffix':FORMAT,'glossary':GLOSSARY,'unknown_node_scoring':'unmapped label is a unique incorrect node, counted in set F1 denominator','task_index_route':{'parents':'IN','bfs':'OUT'}};save(OUT/'frozen.json',fr);client.OUT=OUT;client.TOKENIZER=tok;rs=[]
    for i,p in enumerate(ps):
        for j,seed in enumerate(fr['seeds']):
            for arm in fr['orders'][i][j]:
                enc,mode=arm.rsplit('_',1);inv=p['inverse']['original' if enc=='original' else 'compact'];r={'model':client.MODEL,'row_index':p['row_index'],'arm':arm,'repeat':j,'seed':seed,**client.call(p['prompts'][enc],mode=='on',seed)};r['score']=restore_score(r['raw'],p['answer_nodes'],inv,r['error']);rs.append(r);save(OUT/'results.json',rs);print(i+1,p['row_index'],arm,j,r['raw'].get('done_reason'),r['score'],flush=True)
                if not r['error']:assert r['api_input_matches'],'Input calibration mismatch preserved; stop without overwrite.'
    print('Complete96 source-representation x mode calls; eight graph units.',flush=True)
if __name__=='__main__':main()
