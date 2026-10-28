"""Compile manually specified development relations with exact provenance."""
import json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/qasper_relation_ledger'
TRAIN=ROOT/'data/qasper_official/qasper-train-v0.3.json';TEST=ROOT/'data/qasper_official/qasper-test-v0.3.json'

def sha(b):return hashlib.sha256(b).hexdigest()

def paragraphs(raw):
    return [{'heading':'Abstract','text':raw['abstract']}]+[{'heading':s['section_name'],'text':t.strip()} for s in raw['full_text'] for t in s['paragraphs'] if t.strip()]

def pointer(raw,anchor):
    ps=paragraphs(raw);matches=[(i,p,p['text'].find(anchor)) for i,p in enumerate(ps) if anchor in p['text']]
    assert len(matches)==1,(anchor,len(matches))
    i,p,a=matches[0];text=p['text'];b=a+len(anchor);byte=len(text[:a].encode());n=len(anchor.encode())
    assert text[a:b]==anchor and text.encode()[byte:byte+n].decode()==anchor
    return {'paragraph_id':i,'heading':p['heading'],'quote':anchor,'char_start':a,'char_length':b-a,'byte_start':byte,'byte_length':n,'paragraph_sha256':sha(text.encode())}

def main():
    OUT.mkdir(parents=True,exist_ok=True);train=json.loads(TRAIN.read_text(encoding='utf-8'));test=json.loads(TEST.read_text(encoding='utf-8'))
    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    specs=[
      ('svm_author','1611.01400','SVMRank','used_method','supervised learning-to-rank','author_method',{},['SVMRank is a modification to SVM that assigns scores to each data point and allows the results to be ranked ( BIBREF26 ). We use SVMRank in the experiments below.']),
      ('crf_prior','1611.01400','Conditional Random Fields','used_method','query generation','cited_prior_work',{},['BIBREF9 employ supervised machine learning model (i.e., Conditional Random Fields) ( BIBREF12 ) for query generation.']),
      ('tarc_direction','2003.09520','TArC semi-automatic procedure','transcription_direction','Arabish to Arabic script','author_method',{},['the semi-automatic procedure concerns only transcription from Arabish to Arabic script.']),
      ('tarc_sequence','2003.09520','TArC sequential model','mapping','Arabish characters to Arabic morphemes','author_method',{},['Our model learns thus to map Arabish characters into Arabic morphemes.']),
      ('tarc_initial','2003.09520','TArC sequential model','token_accuracy','roughly 71%','author_evaluation',{'stage':'initial 10-fold cross-validation'},['The 10-fold cross validation with this setting gave a token-level accuracy of roughly 71%.']),
      ('tarc_later','2003.09520','TArC sequential model','token_accuracy','roughly 80%','author_evaluation',{'stage':'after adding third block, annotating fourth block'},['Adding the third block to the training data and annotating a fourth block with the new trained model gave in contrast an accuracy of roughly 80%.']),
      ('sent2vec_train','1703.02507','Sent2Vec','training_regime','unsupervised','author_method',{},['We propose a new unsupervised model, Sent2Vec, for learning universal sentence embeddings.']),
      ('fastsent_train','1703.02507','FastSent','training_regime','unsupervised','comparison_model',{},['FastSent BIBREF16 is a sentence-level log-linear bag-of-words model. Like SkipThought, it uses adjacent sentences as the prediction target and is trained in an unsupervised fashion.']),
      ('sent2vec_eval','1703.02507','Sent2Vec','downstream_evaluation','supervised classification','author_evaluation',{},['Sentence embeddings are evaluated for various supervised classification tasks as follows.','For the remaining 5 datasets, Sent2Vec embeddings are inferred from input sentences and directly fed to a logistic regression classifier.']),
      ('mdl_class','2002.06053','Minimum Description Length','segmentation_regime','unsupervised compression-based','surveyed_method',{},['Minimum Description Length (MDL) is an unsupervised compression-based word segmentation technique in which words of an unknown language are detected by compressing the text corpus.'])]
    facts=[]
    for fid,pid,subject,relation,value,role,conditions,anchors in specs:
        assert pid in split['fit']
        facts.append({'id':fid,'paper_id':pid,'subject':subject,'relation':relation,'value':value,'role':role,'conditions':conditions,'evidence':[pointer(train[pid],a) for a in anchors],'semantic_label_author':'assistant_manual_source_interpretation','independent_review':False})
    rejects=[
      {'id':'wrong_author','claim':'This paper develops a CRF query-generation model.','label':'reject_scope','support_fact_ids':['svm_author','crf_prior'],'reason':'Prior cited method is not the authors learning-to-rank method.'},
      {'id':'reverse_direction','claim':'The TArC semi-automatic procedure maps Arabic script to Arabish.','label':'reject_direction','support_fact_ids':['tarc_direction','tarc_sequence'],'reason':'Reverse direction is not supported by the stated author procedure.'},
      {'id':'wrong_stage','claim':'The initial 10-fold experiment achieves roughly 80% token accuracy.','label':'reject_condition','support_fact_ids':['tarc_initial','tarc_later'],'reason':'80% belongs to later incremental stage, initial report is71%.'},
      {'id':'training_eval_confusion','claim':'Sent2Vec is trained supervised because it is evaluated on supervised tasks.','label':'reject_relation','support_fact_ids':['sent2vec_train','sent2vec_eval'],'reason':'Downstream supervised evaluation does not change unsupervised representation training.'},
      {'id':'remove_qualifier','claim':'TArC achieves exactly 71% on every annotation block.','label':'reject_scope_precision','support_fact_ids':['tarc_initial','tarc_later'],'reason':'Source says roughly, binds stage, and reports a different later value.'}]
    conflicts=json.loads((ROOT/'logs/qasper_target_scope/diagnostics.json').read_text(encoding='utf-8'))['reference_conflicts'];audit=[]
    for c in conflicts:
        pid=c['paper_id'];raw=test[pid];entry={'paper_id':pid,'question_id':c['id'],'question':c['question'],'original_references':c['references'],'posthoc':True,'independent_review':False,'review_author':'assistant_manual_source_interpretation'}
        if pid=='1802.09059':
            entry.update(status='source_supported_resolution',answer='Yes: GloVe word embeddings initialize the word lookup table.',reason='Do not confuse word embeddings with sense embeddings trained in-model.',evidence=[pointer(raw,'The INLINEFORM7 (stored in the word embeddings lookup table) can be initialized using pre-trained word embeddings; in this work, GloVe vectors are used.')])
        elif pid=='1809.05807':
            entry.update(status='source_supported_resolution',answer='t-test',reason='Positive explicit statement resolves the unanswerable annotation.',evidence=[pointer(raw,'We also show the p-value by comparing the result of 10 random tests for both our model and the state-of-the-art model in the t-test .')])
        elif pid=='1809.00129':
            anchors=['Two feed-forward layers of size 400 with rectified linear units (ReLU; BIBREF9 );','One bi-directional gated recurrent unit (BiGRU; BIBREF10 ) layer with hidden size 200, where the forward and backward hidden states are concatenated and further normalized by layer normalization BIBREF11 .','Two feed-forward layers of hidden size 200 with rectified linear units;','One BiGRU layer with hidden size 100 using the same configuration of the previous BiGRU layer;','Two feed-forward layers of size 100 and 50 respectively with ReLU activation.']
            entry.update(status='source_supported_resolution',answer='2 recurrent BiGRU layers',reason='Typed listed-stack count:6 feed-forward+2 recurrent=8. Question asks recurrent layers, not all listed encoding-stack layers. Output softmax is outside this listed count.',typed_counts={'feed_forward':6,'recurrent':2,'listed_encoding_stack':8},evidence=[pointer(raw,a) for a in anchors])
        elif pid=='2003.13016':
            entry.update(status='unresolved_scope',answer=None,reason='A previously reported HeidelTime score is not proof of separate evaluation of this newly auto-annotated time-expression corpus; NER evaluation is a different annotation type.',evidence=[pointer(raw,'This version of Heideltime achieves an F$_1$ value of 89.1 for partial identification and normalization.'),pointer(raw,'To evaluate the performance we used stratified 10-fold cross-validation.')])
        else:
            entry.update(status='unresolved_scope_absence',answer=None,reason='Pretrained seq2seq model is explicit, but cannot substitute for pretrained embeddings like BERT. No BERT occurrence in this serialized source is not a semantic proof of No.',evidence=[pointer(raw,'To train the fusion model, we first pretrain a Conv seq2seq with self-attention model on the WritingPrompts dataset.')])
        audit.append(entry)
    result={'schema_version':1,'coordinate_system':'abstract raw; stripped nonempty body paragraphs; char=Unicode codepoints, byte=UTF8 relative to paragraph','facts':facts,'reject_examples':rejects,'posthoc_reference_adjudications':audit,'semantic_verifier_implemented':False,'future_eval_requirement':'independent label review and frozen split before calls'}
    out=OUT/'ledger.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    files=[TRAIN,TEST,Path(__file__),ROOT/'protocol_qasper_relation_ledger_20261002.md',ROOT/'logs/qasper_fingerprint/split.json',ROOT/'logs/qasper_target_scope/diagnostics.json',out]
    pointers=sum(len(f['evidence']) for f in facts)+sum(len(a['evidence']) for a in audit)
    summary={'facts':len(facts),'fit_papers':len({f['paper_id'] for f in facts}),'manual_reject_examples':len(rejects),'reference_cases':len(audit),'source_supported_resolutions':sum(a['status']=='source_supported_resolution' for a in audit),'unresolved':sum(a['answer'] is None for a in audit),'verified_quote_pointers':pointers,'ledger_utf8_bytes':len(out.read_bytes()),'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'not_automated_semantic_accuracy':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
