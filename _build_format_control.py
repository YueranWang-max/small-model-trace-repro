"""Create the independent runner from the archived, known replication implementation."""
from pathlib import Path
p=Path(__file__).parent
s=(p/'run_qasper_contract_replication.py').read_text(encoding='utf-8')
s=s.replace('Independent fixed-sample replication of question contracts with one shared parser.', 'Frozen official-test formatting comparator for full question contracts.')
s=s.replace("OUT=ROOT/'logs/qasper_contract_replication'", "OUT=ROOT/'logs/qasper_format_control'")
s=s.replace("SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'", "SOURCE=ROOT/'data/qasper_official/qasper-test-v0.3.json'")
s=s.replace("PROTOCOL=ROOT/'protocol_qasper_contract_replication_20261002.md'", "PROTOCOL=ROOT/'protocol_qasper_format_control_20261002.md'")
s=s.replace("ARMS=['base','contract']", "ARMS=['base','format','contract']")
start=s.index('def contract(text):')
end=s.index('def parsed(result):')
s=s[:start]+'''from run_qasper_contract_replication import contract

def format_rule(kind):
    return {
        'boolean':'The answer VALUE after the Final Answer: prefix must be exactly Yes, No, or unknown.',
        'quantity':'Write a concise numeric answer.',
        'list':'Write a concise comma-separated list.',
        'procedure':'Write a concise sequence of steps.',
        'comparison':'Write a concise phrase or sentence.',
        'fact':'Write a concise phrase.'
    }[kind]

'''+s[end:]
s=s.replace("    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))\n    ids=split['heldout'][24:56]", "    ids=sorted(source,key=lambda pid:sha(pid.encode()))[40:70]\n    prior=json.loads((ROOT/'logs/qasper_demand_transfer/frozen.json').read_text(encoding='utf-8'))\n    assert not set(ids).intersection(prior['reader_paper_ids'])")
s=s.replace("order=frozen['arm_order'] if pi%2==0 else frozen['arm_order'][::-1]", "order=frozen['arm_order'];order=order[pi%3:]+order[:pi%3]")
s=s.replace("                if arm=='contract':prompt+=", "                if arm=='format':prompt+='Answer value format: '+format_rule(choice['contract'][0])+'\\n'\n                if arm=='contract':prompt+=")
s=s.replace("'rescore_qasper_demand_transfer.py']", "'rescore_qasper_demand_transfer.py','run_qasper_contract_replication.py']")
s=s.replace('assert len(groups)==32', 'assert len(groups)==30')
s=s.replace("out={'papers':32", "out={'papers':30")
start=s.index('    rng=random.Random(20261002);boot=[]',s.index('def analyze():'))
end=s.index("    (OUT/'summary.json')",start)
s=s[:start]+'''    out['paired']={}
    for arm,ref in [('contract','format'),('contract','base'),('format','base')]:
        rng=random.Random(20261002);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]]
            boot.append(sum(r['arms'][arm]['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
        boot.sort();out['paired'][arm+'_minus_'+ref]={'difference':out['arms'][arm]['official_f1']-out['arms'][ref]['official_f1'],'paper_ci95':[boot[250],boot[9750]]}
    out['gate_pass']=out['paired']['contract_minus_format']['paper_ci95'][0]>0
'''+s[end:]
(p/'run_qasper_format_control.py').write_text(s,encoding='utf-8')
