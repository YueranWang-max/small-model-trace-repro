"""Single post-hoc source dispute and conditional sensitivity, frozen scores intact."""
import json,random
from run_qasper_attention_reader import OUT,SOURCE,sha,ARMS
from build_qasper_relation_ledger import paragraphs
from rescore_qasper_demand_transfer import f1

def main():
    source=json.loads(SOURCE.read_text(encoding='utf-8'));fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));reader=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));paper=fr['papers'][0];task=paper['questions'][1];assert paper['paper_id']=='1911.11933'
    q=next(q for q in source[paper['paper_id']]['qas'] if q['question_id']==task['id']);ps=paragraphs(source[paper['paper_id']]);quotes=[]
    for i,phrase in [(34,'We used'),(36,'The proposed method reduced')]:
        body=ps[i]['text'];s=next(s for s in paper['units'] if s['pid']==i and s['text'].startswith(phrase));a=s['start'];b=a+s['length'];assert body[a:b]==s['text'];quotes.append({'paragraph_id':i,'heading':ps[i]['heading'],'quote':s['text'],'char_start':a,'char_length':s['length'],'byte_start':len(body[:a].encode()),'byte_length':len(s['text'].encode()),'paragraph_sha256':sha(body.encode())})
    row=next(r for r in reader['rows'] if r['id']==task['id']);dispute={'paper_id':paper['paper_id'],'question_id':task['id'],'question':q['question'],'original_annotations':q['answers'],'source_quotes':quotes,'assistant_source_interpretation':'Yes: general NMT models are baselines and performance is compared against a full-sentence baseline. This conflicts with the public No label. Independent adjudication pending.','independently_adjudicated':False,'primary_labels_and_scores_unchanged':True,'predictions':{a:row['arms'][a]['answer'] for a in ARMS}}
    (OUT/'label_dispute.json').write_text(json.dumps(dispute,indent=2,ensure_ascii=False),encoding='utf-8');scores={r['id']:{a:(f1(r['arms'][a]['answer'],'Yes') if r['id']==task['id'] else r['arms'][a]['official_f1']) for a in ARMS} for r in reader['rows']};means={a:sum(s[a] for s in scores.values())/len(scores) for a in ARMS};groups=[[r['id'] for r in reader['rows'] if r['paper_id']==p['paper_id']] for p in fr['papers']];pairs={}
    for a,b in [('question_attention','full'),('question_attention','elements'),('shared_attention','elements')]:
        rng=random.Random(20261002);bs=[]
        for _ in range(10000):
            sample=[i for _ in groups for i in rng.choice(groups)];bs.append(sum(scores[i][a]-scores[i][b] for i in sample)/12)
        bs.sort();pairs[a+'_minus_'+b]={'diff':means[a]-means[b],'ci95':[bs[250],bs[9750]]}
    out={'label':'Conditional post-hoc sensitivity IF this one source interpretation is independently confirmed; NOT revised gold or primary results','changed_question_id':task['id'],'conditional_reference':'Yes','means':means,'paired':pairs,'original_reader_sha256':sha((OUT/'reader.json').read_bytes()),'single_question_only':True,'others_not_revalidated':True};(OUT/'label_sensitivity.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
