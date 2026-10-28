"""Paired Qasper readout with reusable source prefix versus per-paper BM25."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from analyze_qasper_pointer_oracle import serialize_paper
from prepare_qasper_multiq import gold_answers
from repro_lib import ollama_chat
from run_qasper_outline_route import build_paper
from run_realdoc import score
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_prefix_vs_bm25"
MODEL = "qwen3.5:4b"
SYSTEM = "You answer research-paper questions using only supplied evidence. Give a short answer."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ask(material, question, inst):
    prompt = ("Answer using only the evidence below. If insufficient, say unknown.\n"
              "Evidence:\n" + material + "\n\nQuestion: " + question +
              "\nEnd with exactly one line: Final Answer: <short answer>")
    t0 = time.perf_counter()
    try:
        g = ollama_chat(MODEL,SYSTEM,prompt,num_ctx=32768,think=False,temperature=0.0,num_predict=1024)
        answer_lines = re.findall(r"(?im)^Final Answer:\s*(.*?)\s*$",g.content)
        answer = answer_lines[-1].strip() if answer_lines else ""
        f1,em = score([answer] if answer else [],inst)
        raw = g.raw
        return {"answer":answer,"f1":f1,"em":em,"wall_sec":round(time.perf_counter()-t0,3),
                "prompt_chars":len(prompt),"prompt_tokens":raw.get("prompt_eval_count"),
                "generation_tokens":raw.get("eval_count"),
                "prompt_eval_duration_ns":raw.get("prompt_eval_duration"),
                "eval_duration_ns":raw.get("eval_duration"),
                "total_duration_ns":raw.get("total_duration"),
                "load_duration_ns":raw.get("load_duration"),
                "raw_content":g.content,
                "response_sha256":sha(json.dumps(raw,sort_keys=True).encode())}
    except Exception as exc:
        return {"answer":"","f1":0.0,"em":0.0,"wall_sec":round(time.perf_counter()-t0,3),
                "error":repr(exc)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    ranked = sorted(source,key=lambda p:(-len(source[p]["qas"]),p))
    ids = []
    for pid in ranked[105:]:
        n = len(serialize_paper(source[pid]).decode("utf-8"))
        if len(source[pid]["qas"])>=4 and 8000<=n<=32000:
            ids.append(pid)
        if len(ids)==9:
            break
    assert len(ids)==9
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen["paper_ids"]==ids
    else:
        frozen={"source_sha256":sha(DEV.read_bytes()),"paper_ids":ids,
                "question_ids":{pid:[q["question_id"] for q in source[pid]["qas"]] for pid in ids},
                "selection":"rank >=106, first nine with >=4 questions and 8000<=canonical chars<=32000"}
        frozen_path.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status":"exploratory_shared_prefix_full_source_vs_bm25","frozen_sha256":sha(frozen_path.read_bytes()),
        "model":MODEL,"papers":[]}
    assert results["frozen_sha256"]==sha(frozen_path.read_bytes())
    for pi,pid in enumerate(ids):
        paper=build_paper(pid,source[pid])
        raw_source=serialize_paper(source[pid]).decode("utf-8")
        t0=time.perf_counter()
        index=BM25(paper["paragraphs"])
        index_sec=time.perf_counter()-t0
        rp=next((x for x in results["papers"] if x["paper_id"]==pid),None)
        if rp is None:
            rp={"paper_id":pid,"source_chars":len(raw_source),"source_sha256":sha(raw_source.encode()),
                "bm25_index_sec":round(index_sec,6),"arm_order":["source_prefix","bm25"] if pi%2==0 else ["bm25","source_prefix"],
                "records":[]}
            results["papers"].append(rp)
        for arm in rp["arm_order"]:
            for original in source[pid]["qas"]:
                qid=original["question_id"]
                rr=next((x for x in rp["records"] if x["id"]==qid),None)
                if rr is None:
                    rr={"id":qid,"question":original["question"],"arms":{}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                question="Paper: "+paper["title"]+". Question: "+original["question"]
                if arm=="bm25":
                    t0=time.perf_counter()
                    selected=index.search(question,16)
                    rr["bm25_selected"]=selected
                    rr["bm25_search_sec"]=round(time.perf_counter()-t0,6)
                    material="\n\n".join(paper["paragraphs"][i] for i in selected)
                else:
                    material=raw_source
                gold=gold_answers(original)
                inst={"question":question,"answer":gold[0],"answer_aliases":gold[1:]}
                rr["arms"][arm]=ask(material,question,inst)
                path.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
                print(pid,qid,arm,round(rr["arms"][arm]["f1"],3),flush=True)


if __name__=="__main__":
    main()
