"""Post hoc Qasper Counter-token scoring; never replaces the frozen primary gate."""
import collections
import json
import random
import re
import string
from run_qasper_demand_transfer import OUT, SOURCE
from run_qasper_prefix_vs_bm25 import sha

def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    n = sum((collections.Counter(p) & collections.Counter(g)).values())
    return 2*n/(len(p)+len(g)) if n else 0.0

def references(q):
    out = []
    for ref in q['answers']:
        a = ref['answer']
        if a['unanswerable']: s = 'Unanswerable'
        elif a['extractive_spans']: s = ', '.join(a['extractive_spans'])
        elif a['free_form_answer']: s = a['free_form_answer']
        elif a['yes_no'] is True: s = 'Yes'
        elif a['yes_no'] is False: s = 'No'
        else: raise ValueError(a)
        out.append(s)
    return out

def main():
    data = json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    src = json.loads(SOURCE.read_text(encoding='utf-8'))
    groups = []
    for paper in data['papers']:
        group = []
        for row in paper['records']:
            q = next(q for q in src[paper['paper_id']]['qas'] if q['question_id'] == row['id'])
            gold = references(q)
            rec = {'id': row['id'], 'arms': {}}
            for arm, result in row['arms'].items():
                pred = result.get('answer') or ''
                aligned = 'Unanswerable' if normalize(pred) == 'unknown' else pred
                rec['arms'][arm] = {'literal_f1': max(f1(pred,g) for g in gold),
                    'prompt_aligned_f1': max(f1(aligned,g) for g in gold),
                    'normalized_text_em': float(any(normalize(aligned)==normalize(g) for g in gold))}
            group.append(rec)
        groups.append(group)
    assert len(groups)==40 and all(set(r['arms'])=={'bm25','global'} for g in groups for r in g)
    rows = [r for g in groups for r in g]
    summary = {}
    for metric in ['literal_f1','prompt_aligned_f1','normalized_text_em']:
        means = {a: sum(r['arms'][a][metric] for r in rows)/len(rows) for a in ['bm25','global']}
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sample = [r for _ in groups for r in groups[rng.randrange(len(groups))]]
            boot.append(sum(r['arms']['global'][metric]-r['arms']['bm25'][metric] for r in sample)/len(sample))
        boot.sort()
        summary[metric] = {**means,'difference':means['global']-means['bm25'],'paper_ci95':[boot[250],boot[9750]]}
    output = {'label':'post hoc audit; frozen primary gate unchanged','papers':40,'questions':len(rows),
        'source':'https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py',
        'policy':'Counter token F1 and official normalization. Literal uses raw final answer; prompt-aligned converts only exact normalized unknown to Unanswerable. Text EM is supplemental, not an official Qasper metric.',
        'reader_results_sha256':sha((OUT/'reader_results.json').read_bytes()),'summary':summary,'groups':groups}
    (OUT/'reader_official_rescore.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
