"""Frozen, lossless node-name representation probe; no answer-derived mapping."""
import itertools,json,random,re,time
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import run_official_graph_sampling_control as client
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,OPENER,sha,save,score,oracle
from run_official_graph_five_trace import FORMAT
OUT=ROOT/'logs/official_graph_identity_recode'
PROTOCOL=ROOT/'protocol_official_graph_identity_recode_20261003.md'
HASH=re.compile(r'\b[0-9a-f]{10}\b')
ARMS=['original','compact','long_alias']
def recode(prompt,row_index,tokenizer):
    started=time.perf_counter();ids=sorted(set(HASH.findall(prompt)))
    source_words=set(re.findall(r'\b[A-Z]{1,3}\b',prompt))
    labels=sorted([s for s in tokenizer.get_vocab() if s not in source_words and re.fullmatch('[A-Z]{1,3}',s) and len(tokenizer.encode(s,add_special_tokens=False).ids)==1],key=lambda s:(len(s),s))[:len(ids)]
    assert len(labels)==len(ids)
    random.Random(202610063+row_index).shuffle(labels)
    short=dict(zip(ids,labels));long={};seen=set(ids)
    for s in ids:
        j=0
        while True:
            a=sha(f'identity-recode-20261003:{row_index}:{s}:{j}'.encode())[:10]
            if a not in seen:break
            j+=1
        seen.add(a);long[s]=a
    maps={'original':{s:s for s in ids},'compact':short,'long_alias':long}
    prompts={arm:HASH.sub(lambda m:mp[m[0]],prompt) for arm,mp in maps.items()}
    inverse={arm:{v:k for k,v in mp.items()} for arm,mp in maps.items()}
    for arm in ARMS:
        pat=HASH if arm!='compact' else re.compile(r'\b(?:'+ '|'.join(sorted(inverse[arm],key=len,reverse=True))+r')\b')
        assert pat.sub(lambda m:inverse[arm][m[0]],prompts[arm])==prompt
    return {'maps':maps,'inverse':inverse,'prompts':prompts,'preprocess_wall_sec':time.perf_counter()-started,'dictionary_json_bytes':len(json.dumps({'maps':maps,'inverse':inverse},sort_keys=True).encode()),'compact_dictionary_json_bytes':len(json.dumps({'map':short,'inverse':inverse['compact']},sort_keys=True).encode())}
def mapped_score(raw,gold,inverse,error=None):
    s=score(raw,[],error);p=s['prediction'];known=all(v in inverse for v in p)
    mapped=[inverse[v] for v in p if v in inverse];a,b=set(mapped),set(gold);valid=s['valid'] and known
    f=2*len(a&b)/(len(a)+len(b)) if a or b else 1.
    return {'prediction':p,'restored_prediction':mapped,'unknown_prediction_nodes':[v for v in p if v not in inverse],'valid':valid,'em':float(valid and a==b),'f1':f if valid else 0.}
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_sampling_control.py','run_official_graph_long_boundary.py','run_official_graph_native_probe.py','run_official_graph_five_trace.py','data/qwen3_tokenizer_official/tokenizer.json']}
    assert not (OUT/'frozen.json').exists() and not (OUT/'results.json').exists() and not list(OUT.glob('attempt_*.json'))
    used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(p['prompt'].rsplit('Operation:',1)[0].encode()))
    rows=pq.read_table(DATA).to_pylist();ps=[];tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(rows):
                h=sha(p['prompt'].rsplit('Operation:',1)[0].encode())
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or h in used:continue
                used.add(h);q={'row_index':i,'graph_head_sha256':h,'length_bin':[lo,hi],**p};q.update(recode(p['prompt']+'\n'+FORMAT,i,tok))
                assert set(oracle(p))==set(p['answer_nodes'])
                for arm in ARMS:
                    out=oracle({'prompt':q['prompts'][arm],'problem_type':typ})
                    assert {q['inverse'][arm][n] for n in out}==set(p['answer_nodes'])
                ps.append(q);count+=1
                if count==2:break
            assert count==2
    ps=[ps[j] for j in [0,2,4,6,1,3,5,7]]
    with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
    identity=next(m for m in inventory['models'] if m['name']=='qwen3:1.7b');assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
    orders=list(itertools.permutations(ARMS));seeds=[202610060,202610061]
    fr={'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':ps,'identity':identity,'model_inventory':inventory,'regime':client.REGIMES['recommended'],'ctx':32768,'cap':8192,'seeds':seeds,'orders':[[list(orders[(2*i+j)%6]) for j in range(2)] for i in range(8)],'selection':'first2 per type per bin, graph-head disjoint six earlier stages','format_suffix':FORMAT}
    save(OUT/'frozen.json',fr);records=[];client.OUT=OUT;client.CTX=32768;client.CAP=8192
    for i,p in enumerate(ps):
        for j,seed in enumerate(seeds):
            for arm in fr['orders'][i][j]:
                r={'model':'qwen3:1.7b','row_index':p['row_index'],'arm':arm,'repeat':j,'seed':seed,**client.call(p['prompts'][arm],'recommended',seed)}
                r['score']=mapped_score(r['raw'],p['answer_nodes'],p['inverse'][arm],r['error']);records.append(r);save(OUT/'results.json',records);print(i+1,p['row_index'],arm,j,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete48 calls; eight independent graphs with paired two seeds.',flush=True)
if __name__=='__main__':main()
