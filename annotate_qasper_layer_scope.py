"""Posthoc assistant scope inspection; not an independent semantic gold set."""
import json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/qasper_layer_mentions'
NOTES=[
 ('block_scope','DPCNN shortcut/skipped-layer block','Negation concerns dimension matching, not the existence of two convolutional layers; blanket sentence guard over-rejects.'),
 ('shared_component_statement','HR-VAE encoder and decoder','Two-layer LSTM is stated for both; preserve both components, do not sum into a four-layer single component.'),
 ('component_scope','character embedder','First mention belongs to character embedder, distinct from tagger.'),
 ('component_scope','tagger','Second mention belongs to tagger, distinct from character embedder.'),
 ('named_function_scope','g','Feed-forward network g; not the entire NMT model.'),
 ('named_function_scope','r','Alignment network r; do not merge with g.'),
 ('counting_convention_ambiguous','Bi-LSTM internal structure','Two LSTM directions/components do not automatically mean two stacked BiLSTM layers.'),
 ('per_block_scope','each FCN layer','Each of three FCN layers contains two convolutional layers; local two is per-block, not the global count.'),
 ('component_scope','encoder','Explicit encoder has two LSTM layers.'),
 ('component_scope','decoder','Explicit decoder also has two LSTM layers.'),
 ('classifier_scope','CNN classifier for attention maps','Eight convolutional layers belong to classifier, not BERT whose maps are analyzed.'),
 ('shared_component_ambiguous','LSTM Encoder/Decoder','Slash notation alone does not fully resolve per-component versus shared depth conventions.'),
 ('prefix_subset_scope','first two LSTM layers','First two is a subset marker, not proof of total depth two.'),
 ('task_scope','sentiment analysis network','One LSTM layer for this task; other layers also exist.'),
 ('task_scope','character-level language model','One LSTM layer for a different task; not duplicate evidence of same model.'),
 ('named_function_scope','f_pre','Two feed-forward layers in f_pre; surrounding co-attention layer is distinct.'),
 ('named_function_scope','f_post','Two feed-forward layers in f_post; not f_pre.'),
 ('baseline_scope','baseline model BiGRU stack','Three BiGRU layers appear in baseline section; do not automatically assign to proposed variant.'),
 ('direction_convention_ambiguous','bidirectional LSTM','Two-layer LSTM for both directions; retain convention before total arithmetic.'),
 ('operation_count_context','LSTM layers in benchmark','Two LSTM layers are mentioned while computing operation cost; model/task binding still needed.')]

def main():
    p=OUT/'validation_records.json';records=json.loads(p.read_text(encoding='utf-8'));assert len(records)>=20
    inspected=[]
    for i,(kind,subject,note) in enumerate(NOTES):
        r=records[i]
        inspected.append({'source_record_index':i,'paper_id':r['paper_id'],'paragraph_id':r['paragraph_id'],'quote':r['quote'],'context':r['context'],'local_count':r['count'],'scope_type':kind,'proposed_subject':subject,'manual_note':note,'answer_ready_certified':False,'label_author':'assistant_posthoc_source_inspection','independent_review':False})
    result={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'policy':'fixed first20 validation records; inspection after extraction, not used to alter frozen parser','records':inspected,'independent_semantic_accuracy':None}
    out=OUT/'validation_scope_inspection.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print({'inspected':len(inspected),'answer_ready_certified':0})

if __name__=='__main__':main()
