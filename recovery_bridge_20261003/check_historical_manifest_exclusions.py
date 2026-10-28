"""Lightweight archived-input comparison, independent of model outcomes."""
import json,pathlib,re,hashlib,collections
ROOT=pathlib.Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def graph_signature(prompt):
    marker='Here is the graph to operate on:'
    assert prompt.count(marker)==1
    body=prompt.split(marker,1)[1].split('Operation:',1)[0]
    edges=re.findall(r'^([a-z0-9]+) -> ([a-z0-9]+)\s*$',body,re.M)
    assert edges
    # Sorting preserves multiplicity, ignores harmless line/order differences.
    return sha(json.dumps(sorted(edges),separators=(',',':')).encode())
def main():
    out=ROOT/'historical_manifest_exclusion_check_20261004.json'
    assert not out.exists(),'Do not overwrite archived check.'
    historical={};archives=[]
    for path in sorted((ROOT.parent/'logs').glob('official_graph_*/frozen.json')):
        raw=path.read_bytes();j=json.loads(raw)
        rows=j.get('selected',j.get('cases',[]));indices=[]
        for record in rows:
            case=record.get('paper_row',record)
            index=case['row_index'];signature=graph_signature(case['prompt'])
            assert index not in historical or historical[index]==signature
            historical[index]=signature;indices.append(index)
        archives.append({'path':str(path),'sha256':sha(raw),'selected_row_indices':indices})
    expected=set(range(16))|set(range(100,113))|set(range(400,416))|set(range(450,463))
    assert set(historical)==expected,'Session-reconstructed exclusions differ from archived manifests.'
    frozenraw=(ROOT/'bridge_run/frozen.json').read_bytes();bridge=json.loads(frozenraw)
    signatures=[graph_signature(c['prompt']) for c in bridge['cases']]
    historical_signatures=set(historical.values())
    overlap=[c['row_index'] for c,s in zip(bridge['cases'],signatures) if s in historical_signatures]
    checks={'historical_row_set_matches_session_reconstruction':set(historical)==expected,
        'bridge_row_indices_disjoint':not (set(c['row_index'] for c in bridge['cases'])&expected),
        'bridge_canonical_edge_multisets_disjoint':not overlap,
        'bridge_all_16_graphs_distinct':len(set(signatures))==16}
    result={'checks':checks,'all_checks_pass':all(checks.values()),'archives':archives,
        'historical_rows':sorted(historical),'historical_distinct_edge_multisets':len(historical_signatures),
        'bridge_frozen_sha256':sha(frozenraw),'bridge_rows':[c['row_index'] for c in bridge['cases']],
        'bridge_graph_signatures':signatures,'overlap_rows':overlap,
        'scope':'Post-freeze comparison to ten restored official-study manifests; no scoring or inference.',
        'limitations':['Does not establish absence from unrelated or unarchived experiments.',
            'Does not alter the original pre-inference historical_manifest_verification pending record.',
            'Does not replace dataset label, request or response qualification audits.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['archives','bridge_graph_signatures']},indent=2))
    assert all(checks.values())
if __name__=='__main__':main()
