"""Compare archived question metadata to current-generator replays; no inference."""
import pathlib,json,hashlib,random,re,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from repro_lib import sample_from_target_chars
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def independent_gold(edges,operation,kind):
    if kind=='parents':
        target=re.search(r'parents of node (n\d+)',operation)[1]
        return {u for u,v in edges if v==target}
    root,depth=re.search(r'BFS from node (n\d+) with depth (\d+)',operation).groups()
    seen={root};frontier={root}
    for _ in range(int(depth)):
        frontier={v for u,v in edges if u in frontier and v not in seen};seen.update(frontier)
    return frontier
def main():
    out=ROOT/'paper/legacy_graph_reconstruction_20261004.json'
    assert not out.exists(),'Preserve existing reconstruction evidence.'
    specs=[('chain_2b_3k.json',7),('chain_2b_3k_s13.json',13)]
    groups=[];signatures=[]
    for name,seed in specs:
        path=ROOT/'logs'/name;archive=json.loads(path.read_bytes())
        rng=random.Random(seed);cases=[]
        for index,record in enumerate(archive['records']):
            kind='bfs' if index%2==0 else 'parents'
            sample=sample_from_target_chars(rng,archive['summary']['target_chars'],op_type=kind)
            checks={key:getattr(sample,key)==record[key] for key in ['op_type','op_text','answer','graph_chars']}
            checks['reconstructed_gold_independently_correct']=independent_gold(sample.edges,sample.op_text,kind)==set(sample.answer)
            signature=hashlib.sha256(json.dumps(sorted(sample.edges),separators=(',',':')).encode()).hexdigest()
            signatures.append(signature)
            cases.append(dict(index=index,checks=checks,reconstructed_graph_sha256=signature,edge_count=len(sample.edges),
                operation=sample.op_text,answer_size=len(sample.answer)))
        groups.append(dict(path=str(path),sha256=sha(path),seed=seed,seed_provenance='Current default plus full question-metadata replay consistency' if seed==7 else 'Archived scaleup_chain_s13.log header plus replay consistency',
            records=len(cases),all_record_metadata_match=all(all(c['checks'].values()) for c in cases),cases=cases))
    ref=json.loads((ROOT/'logs/chain_2b_3k_s13.json').read_bytes())['records']
    replay_path=ROOT/'logs/chain2_hard_2b.json';replay=json.loads(replay_path.read_bytes())['records']
    duplicate_checks=[]
    for record in replay:
        index=record['orig_index'];original=ref[index]
        checks={k:record[k]==original[k] for k in ['op_text','answer','graph_chars']}
        checks['reference_conditions_are_exact_copies']=record['ref_conditions']==original['conditions']
        checks['reference_first_mean_matches']=record['ref_first_mean']==sum(original['first_f1s'])/len(original['first_f1s'])
        duplicate_checks.append(dict(orig_index=index,checks=checks))
    result=dict(scope='Current-generator reconstruction compared to archived metadata; no historical edge archive recovered.',
        source_files={str(ROOT/name):sha(ROOT/name) for name in ['repro_lib.py','run_chain.py','run_chain2.py']},
        runtime_python=sys.version,groups=groups,reconstructed_records=len(signatures),reconstructed_distinct_graphs=len(set(signatures)),
        replay_archive_sha256=sha(replay_path),replay_reference_checks=duplicate_checks,
        replay_reference_indices=[r['orig_index'] for r in replay],
        all_checks_pass=all(g['all_record_metadata_match'] for g in groups) and all(all(r['checks'].values()) for r in duplicate_checks),
        actual_archived_unique_graph_count_verified=False,historical_code_identity_verified=False,historical_strict_scores_recomputed=False,
        inference_calls=0,limitations=['Matching operation, answer and character count is strong replay consistency evidence, but does not prove historical edge identity without original edge snapshots.',
            'Replayed BFS reference records are not additional independent cases.',
            'Generated edges are reconstructions, not recovered historical raw input.',
            'No visible answers or termination metadata recovered; strict rescoring remains unavailable from these archives.'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','replay_reference_checks']},ensure_ascii=False,indent=2))
    assert result['all_checks_pass']
if __name__=='__main__':main()
