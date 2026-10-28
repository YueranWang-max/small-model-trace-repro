"""Genre-conditioned functional navigation and question-focused extractive summaries."""
from __future__ import annotations
import json
import re
import random
import time
import argparse
from run_qasper_prefix_vs_bm25 import DEV, ROOT, sha
from run_stageC import BM25
from analyze_qasper_evidence_gaps import norm

OUT = ROOT / "logs/qasper_genre_structure"
ROLES = {
    "background": r"\b(introduction|background|motivation|objective|purpose|goal|aim|problem)\b",
    "method": r"\b(method|methods|methodology|approach|model|architecture|algorithm|framework|training|implementation|component|components|module|modules)\b",
    "data": r"\b(data|dataset|datasets|corpus|corpora|annotation|annotated|collection|sample|samples|language|languages)\b",
    "evaluation": r"\b(evaluation|evaluate|evaluated|metric|metrics|baseline|baselines|benchmark|benchmarks|experiment|experiments|setup)\b",
    "results": r"\b(result|results|performance|improvement|improve|outperform|accuracy|score|scores|comparison|compare|compared|better|worse|ablation)\b",
    "related": r"\b(related|previous|prior|existing|literature|citation|cited)\b",
    "conclusion": r"\b(conclusion|conclusions|discussion|limitation|limitations|future|drawback|drawbacks)\b",
}
ARMS = ("flat8", "structure8", "flat_summary", "structure_summary")

def roles(text):
    return sorted(k for k, pattern in ROLES.items() if re.search(pattern, text.lower()))

def body_map(raw):
    paragraphs = [{"text": raw["abstract"], "heading": "Abstract", "roles": ["background", "method", "results", "conclusion"]}]
    for sec in raw["full_text"]:
        heading = sec["section_name"] or ""
        for text in sec["paragraphs"]:
            if text.strip():
                inferred = roles(heading)
                if not inferred:
                    inferred = roles(re.split(r"(?<=[.!?])\s+", text.strip())[0])
                paragraphs.append({"text": text.strip(), "heading": heading, "roles": inferred})
    return paragraphs

def make_summary(units, rank, budget=6000):
    selected = []
    used = 0
    for i in rank:
        u = units[i]
        line = f"[P{u['pid']}] {u['text']}"
        cost = len(line) + (1 if selected else 0)
        if used + cost <= budget:
            selected.append(i)
            used += cost
    selected.sort()
    text = "\n".join(f"[P{units[i]['pid']}] {units[i]['text']}" for i in selected)
    assert len(text) <= budget
    return {"unit_ids": selected, "text": text, "chars": len(text)}

def freeze(source, start=187, count=30):
    ranked = sorted(source, key=lambda p: (-len(source[p]["qas"]), p))
    papers = []
    for pid in ranked[start-1:start-1+count]:
        construction_start = time.perf_counter()
        paras = body_map(source[pid])
        index = BM25([p["text"] for p in paras])
        units = []
        for pi, p in enumerate(paras):
            for sentence in re.split(r"(?<=[.!?])\s+", p["text"]):
                if sentence.strip():
                    units.append({"pid": pi, "text": sentence.strip(), "roles": p["roles"]})
        sindex = BM25([u["text"] for u in units])
        qs = []
        for q in source[pid]["qas"]:
            query = q["question"]
            qr = roles(query)
            rank = index.search(query, len(paras))
            base = rank[:8]
            structured = rank[:6]
            for i in rank:
                if len(structured) >= len(base):
                    break
                if i not in structured and set(qr) & set(paras[i]["roles"]):
                    structured.append(i)
            for i in rank:
                if len(structured) >= len(base):
                    break
                if i not in structured:
                    structured.append(i)
            srank = sindex.search(query, len(units))
            compatible = [i for i in srank if set(qr) & set(units[i]["roles"])]
            cpos = {i: j for j, i in enumerate(compatible)}
            gpos = {i: j for j, i in enumerate(srank)}
            enhanced = sorted(srank, key=lambda i: -(1 / (60 + gpos[i]) +
                                (0.5 / (60 + cpos[i]) if i in cpos else 0))) if qr else srank
            qs.append({"id": q["question_id"], "question": query, "roles": qr,
                       "flat8": base, "structure8": structured,
                       "flat_summary": make_summary(units, srank),
                       "structure_summary": make_summary(units, enhanced)})
        papers.append({"paper_id": pid, "genre": "research_article", "paragraphs": paras,
                       "units": units, "questions": qs,
                       "construction_and_selection_sec": time.perf_counter() - construction_start,
                       "source_sha256": sha(json.dumps(paras, sort_keys=True).encode())})
        print(pid, len(qs), flush=True)
    return {"dataset_sha256": sha(DEV.read_bytes()), "selection": f"dev ranks{start}-{start+count-1} inclusive",
            "rules": ROLES, "papers": papers}

def candidates(q):
    pieces = []
    for annotation in q["answers"]:
        a = annotation["answer"]
        pieces.extend(s for s in (a.get("highlighted_evidence") or a.get("evidence", [])) if len(norm(s)) >= 30)
    return sorted(set(norm(s) for s in pieces))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=187)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", default="logs/qasper_genre_structure")
    args = parser.parse_args()
    global OUT
    OUT = ROOT / args.output
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    fp = OUT / "frozen.json"
    if fp.exists():
        plan = json.loads(fp.read_text(encoding="utf-8"))
        assert plan["dataset_sha256"] == sha(DEV.read_bytes()) and plan["rules"] == ROLES
        assert plan["selection"] == f"dev ranks{args.start}-{args.start+args.count-1} inclusive"
    else:
        plan = freeze(source, args.start, args.count)
        fp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    groups = []
    for p in plan["papers"]:
        rows = []
        for q in p["questions"]:
            original = next(x for x in source[p["paper_id"]]["qas"] if x["question_id"] == q["id"])
            spans = candidates(original)
            mapped = [s for s in spans if any(s in norm(v["text"]) for v in p["paragraphs"])]
            eids = {i for i, v in enumerate(p["paragraphs"]) if any(s in norm(v["text"]) for s in mapped)}
            metrics = {}
            for a in ARMS:
                if a.endswith("8"):
                    selected = set(q[a])
                    hits = len(selected & eids)
                    metrics[a] = {"eligible": bool(eids), "recall": hits / len(eids) if eids else 0,
                        "any": bool(hits), "all": bool(eids) and hits == len(eids),
                        "chars": sum(len(p["paragraphs"][i]["text"]) for i in selected)}
                else:
                    text = norm(q[a]["text"])
                    hits = sum(s in text for s in mapped)
                    metrics[a] = {"eligible": bool(mapped), "recall": hits / len(mapped) if mapped else 0,
                        "any": bool(hits), "all": bool(mapped) and hits == len(mapped), "chars": q[a]["chars"]}
            rows.append({"id": q["id"], "roles": q["roles"], "metrics": metrics})
        groups.append({"paper_id": p["paper_id"], "records": rows})
    rows = [q for p in groups for q in p["records"]]
    summary = {"papers": len(groups), "questions": len(rows), "arms": {}, "paired": {}}
    for a in ARMS:
        vals = [r["metrics"][a] for r in rows]
        eligible = [v for v in vals if v["eligible"]]
        summary["arms"][a] = {"eligible": len(eligible), "any_hit": sum(v["any"] for v in eligible),
            "all_hit": sum(v["all"] for v in eligible),
            "recall": sum(v["recall"] for v in eligible) / len(eligible),
            "chars": sum(v["chars"] for v in vals)}
    for a, b in (("structure8", "flat8"), ("structure_summary", "flat_summary")):
        def delta(papers):
            eligible = [r for p in papers for r in p["records"] if r["metrics"][a]["eligible"]]
            return sum(r["metrics"][a]["recall"] - r["metrics"][b]["recall"] for r in eligible) / len(eligible)
        rng = random.Random(20261003)
        boot = sorted(delta([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(10000))
        summary["paired"][a] = {"recall_difference": delta(groups), "paper_ci95": [boot[250], boot[9750]],
            "gate_pass": boot[250] > 0 and summary["arms"][a]["any_hit"] >= summary["arms"][b]["any_hit"]}
    summary["unrouted_questions"] = sum(not r["roles"] for r in rows)
    summary["question_role_counts"] = {role: sum(role in r["roles"] for r in rows) for role in ROLES}
    summary["full_source_chars_per_question"] = sum(sum(len(v["text"]) for v in p["paragraphs"]) * len(p["questions"]) for p in plan["papers"])
    summary["frozen_sha256"] = sha(fp.read_bytes())
    (OUT / "results.json").write_text(json.dumps({"papers": groups}, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
