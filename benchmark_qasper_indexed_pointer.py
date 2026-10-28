"""Indexed-pointer supplement: avoid reconstructing paragraph arrays per lookup."""
import json,time,statistics,hashlib
from pathlib import Path
from build_qasper_relation_ledger import paragraphs

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/qasper_typed_contract'

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    files=[Path(__file__),ROOT/'logs/qasper_relation_ledger/ledger.json',ROOT/'data/qasper_official/qasper-train-v0.3.json',ROOT/'logs/qasper_typed_contract/frozen.json']
    fp=OUT/'indexed_benchmark_frozen.json';assert not fp.exists();fp.write_text(json.dumps({'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'purpose':'remove avoidable source reserialization from pointer timing; original benchmark preserved','repetitions':5,'lookups_per_repetition':10000,'semantic_accuracy_not_measured':True},indent=2),encoding='utf-8')
    source=json.loads(files[2].read_text(encoding='utf-8'));facts=json.loads(files[1].read_text(encoding='utf-8'))['facts'];start=time.perf_counter()
    index={pid:paragraphs(source[pid]) for pid in {f['paper_id'] for f in facts}}
    encoded={(pid,i):(p['text'],p['text'].encode(),sha(p['text'].encode())) for pid,ps in index.items() for i,p in enumerate(ps)}
    build=time.perf_counter()-start;timings={};outputs={}
    # All lookups are ten already-interpreted records on immutable indexed sources.
    for mode in ('direct_value','indexed_pointer_cached_hash','indexed_pointer_rehash'):
        runs=[]
        for rep in range(5):
            start=time.perf_counter();values=[]
            for _ in range(1000):
                for f in facts:
                    if mode!='direct_value':
                        for e in f['evidence']:
                            text,b,h=encoded[(f['paper_id'],e['paragraph_id'])]
                            checked=sha(b) if mode=='indexed_pointer_rehash' else h
                            assert checked==e['paragraph_sha256']
                            assert text[e['char_start']:e['char_start']+e['char_length']]==e['quote']
                            assert b[e['byte_start']:e['byte_start']+e['byte_length']]==e['quote'].encode()
                    values.append(f['value'])
            runs.append(time.perf_counter()-start)
        outputs[mode]=sha(json.dumps(values,ensure_ascii=False).encode());timings[mode]={'median_sec_10000':statistics.median(runs),'runs_sec':runs}
    assert len(set(outputs.values()))==1
    out={'index_build_sec':build,'timings':timings,'outputs_identical':True,'cached_hash_requires_immutable_source':'On updates invalidate source version/index; cached hashes do not detect external mutation by themselves.','source_data_load_and_manual_interpretation_not_timed':True,'not_network_model_latency':True}
    (OUT/'indexed_benchmark.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
