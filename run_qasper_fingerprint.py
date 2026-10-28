"""Train-only structural fingerprints and demand-conditioned evidence-location priors."""
from __future__ import annotations
import json
import re
import time
import random
from pathlib import Path
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from run_qasper_prefix_vs_bm25 import ROOT, sha
from run_qasper_genre_structure import candidates
from analyze_qasper_evidence_gaps import norm
from run_stageC import BM25

OUT = ROOT / "logs/qasper_fingerprint"
SOURCE = ROOT / "data/qasper_official/qasper-train-v0.3.json"
ZONES = ("abstract", "background", "method", "data", "evaluation", "results", "related", "conclusion", "acknowledgments", "other")
ZONE_RULES = {
    "acknowledgments": r"acknowledg|funding",
    "data": r"dataset|corpus|corpora|annotation|\bdata\b",
    "related": r"related|prior work|previous work|literature",
    "conclusion": r"conclu|limitation|future work|discussion",
    "results": r"result|analysis|comparison|ablation",
    "evaluation": r"evaluat|experiment|baseline|benchmark|setup",
    "background": r"introduc|background|motivat|preliminar",
    "method": r"method|approach|model|architecture|algorithm|framework|training|component|pipeline|system|implementation",
}
DEMAND_RULES = (
    ("provenance", r"\bwhere\b|\b(source|sources|origin|obtained|obtain|collected|collect)\b"),
    ("coverage", r"\b(support|supports|supported|cover|covers|coverage|involve|involves|available|include|includes|contain|contains)\b"),
    ("quantity", r"\bhow (many|much)\b|\b(number|size|amount|percentage|percent)\b"),
    ("baseline", r"\bbaselines?\b|compared? (against|with|to)"),
    ("metric", r"\bmetrics?\b|\bmeasure[ds]?\b|\bcriteria\b"),
    ("limitation", r"limitation|drawback|weakness|future"),
    ("prior", r"previous|prior|related work|cited|citation"),
    ("outcome", r"result|perform|improv|outperform|accuracy|score|better|worse"),
    ("procedure", r"\bhow\b|method|approach|algorithm|train|annotat|architecture|implement"),
)
DEMANDS = tuple(d for d, _ in DEMAND_RULES) + ("generic",)
CUES = ("we propose", "we present", "we evaluate", "our results", "we find", "in contrast", "however", "future work", "outperform", "based on")
KS = (1, 8, 32, 64, 200)
WEIGHTS = (0, 0.1, 0.25, 0.5, 1)

def demand(q):
    return next((d for d, pattern in DEMAND_RULES if re.search(pattern, q.lower())), "generic")

def prepare(pid, raw):
    paras = [{"text": raw["abstract"], "heading": "Abstract", "zone": "abstract"}]
    for sec in raw["full_text"]:
        heading = sec["section_name"] or ""
        tail = heading.split(":::")[-1].lower()
        z = next((z for z, pattern in ZONE_RULES.items() if re.search(pattern, tail)), "other")
        for text in sec["paragraphs"]:
            if text.strip():
                paras.append({"text": text.strip(), "heading": heading, "zone": z})
    n = len(paras)
    cells = [ZONES.index(p["zone"]) * 5 + min(4, int(i / n * 5)) for i, p in enumerate(paras)]
    freq = np.zeros(len(ZONES))
    locations = [[] for _ in ZONES]
    trans = np.zeros((len(ZONES), len(ZONES)))
    zi = [ZONES.index(p["zone"]) for p in paras]
    for i, z in enumerate(zi):
        freq[z] += 1 / n
        locations[z].append(i / n)
        if i:
            trans[zi[i-1], z] += 1 / max(1, n-1)
    text = "\n".join(p["text"].lower() for p in paras)
    feature = list(freq) + [float(np.mean(v)) if v else -1 for v in locations] + list(trans.ravel())
    feature += [np.log1p(n), np.log1p(len(raw["full_text"])), len(raw["abstract"]) / max(1, len(text))]
    feature += [text.count(cue) * 1000 / max(1, len(text)) for cue in CUES]
    index = BM25([p["text"] for p in paras])
    questions = [{"id": q["question_id"], "text": q["question"], "demand": demand(q["question"]),
                  "rank": index.search(q["question"], n)} for q in raw["qas"]]
    return {"id": pid, "paragraphs": paras, "cells": cells, "features": feature, "questions": questions}

def evidence(p, q, source):
    raw = next(v for v in source[p["id"]]["qas"] if v["question_id"] == q["id"])
    spans = candidates(raw)
    return {i for i, para in enumerate(p["paragraphs"]) if any(s in norm(para["text"]) for s in spans)}

def fit_prior(papers, clusters, source):
    positives = np.zeros((len(DEMANDS), 50))
    exposure = np.zeros_like(positives)
    cp = np.zeros((max(clusters)+1, len(DEMANDS), 50))
    ce = np.zeros_like(cp)
    for p, c in zip(papers, clusters):
        for q in p["questions"]:
            e = evidence(p, q, source)
            if not e:
                continue
            d = DEMANDS.index(q["demand"])
            for cell in p["cells"]:
                exposure[d, cell] += 1
                ce[c, d, cell] += 1
            for i in e:
                positives[d, p["cells"][i]] += 1 / len(e)
                cp[c, d, p["cells"][i]] += 1 / len(e)
    global_density = (positives.sum(axis=0) + 1) / (exposure.sum(axis=0) + 100)
    dd = (positives + 20 * global_density) / (exposure + 20)
    cd = (cp + 50 * dd[None, :, :]) / (ce + 50)
    return cd

def select(p, q, density, weight):
    rank = q["rank"]
    if weight == 0:
        return rank[:8]
    d = DEMANDS.index(q["demand"])
    pri = sorted(rank, key=lambda i: -density[d, p["cells"][i]])
    pr = {i: j for j, i in enumerate(pri)}
    return sorted(rank, key=lambda i: -(1 / (60 + rank.index(i)) + weight / (60 + pr[i])))[:8]

def metrics(papers, choices, source):
    records = []
    for p in papers:
        qs = []
        for q in p["questions"]:
            e = evidence(p, q, source)
            arms = {}
            for a, selection in choices[p["id"]][q["id"]].items():
                hit = len(e & set(selection))
                arms[a] = {"recall": hit / len(e) if e else None, "any": hit > 0,
                           "all": bool(e) and hit == len(e),
                           "chars": sum(len(p["paragraphs"][i]["text"]) for i in selection)}
            qs.append({"id": q["id"], "demand": q["demand"], "arms": arms})
        records.append({"paper_id": p["id"], "records": qs})
    return records

def recall(records, arm):
    vals = [q["arms"][arm]["recall"] for p in records for q in p["records"] if q["arms"][arm]["recall"] is not None]
    return sum(vals) / len(vals)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    ids = sorted(source, key=lambda pid: sha(pid.encode()))
    a, b = int(len(ids)*0.6), int(len(ids)*0.8)
    splits = {"fit": ids[:a], "validation": ids[a:b], "heldout": ids[b:], "dataset_sha256": sha(SOURCE.read_bytes())}
    (OUT / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    papers = {pid: prepare(pid, source[pid]) for pid in ids}
    print("prepared", len(papers), flush=True)
    fit = [papers[p] for p in splits["fit"]]
    val = [papers[p] for p in splits["validation"]]
    test = [papers[p] for p in splits["heldout"]]
    scaler = StandardScaler().fit([p["features"] for p in fit])
    x = scaler.transform([p["features"] for p in fit])
    configs = []
    models = {}
    for k in KS:
        model = KMeans(n_clusters=k, random_state=20261003, n_init=10).fit(x) if k > 1 else None
        fc = model.labels_ if model is not None else np.zeros(len(fit), dtype=int)
        density = fit_prior(fit, fc, source)
        vc = model.predict(scaler.transform([p["features"] for p in val])) if model is not None else np.zeros(len(val), dtype=int)
        models[k] = (model, density)
        for w in WEIGHTS:
            choices = {p["id"]: {q["id"]: {"candidate": select(p, q, density[c], w)} for q in p["questions"]} for p, c in zip(val, vc)}
            value = recall(metrics(val, choices, source), "candidate")
            configs.append({"k": k, "weight": w, "validation_recall": value})
        print("k", k, "best", max(v["validation_recall"] for v in configs if v["k"] == k), flush=True)
    global_choice = sorted([v for v in configs if v["k"] == 1], key=lambda v: (-v["validation_recall"], v["weight"]))[0]
    fp_choice = sorted([v for v in configs if v["k"] > 1], key=lambda v: (-v["validation_recall"], v["weight"], v["k"]))[0]
    selection = {"global": global_choice, "fingerprint": fp_choice, "validation_configs": configs}
    (OUT / "parameters.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    joblib.dump({"scaler": scaler, "models": models, "selection": selection}, OUT / "model.joblib")
    choices = {p["id"]: {q["id"]: {"bm25": q["rank"][:8]} for q in p["questions"]} for p in test}
    sizes = {}
    for name, cfg in (("global", global_choice), ("fingerprint", fp_choice)):
        model, density = models[cfg["k"]]
        tc = model.predict(scaler.transform([p["features"] for p in test])) if model is not None else np.zeros(len(test), dtype=int)
        for p, c in zip(test, tc):
            for q in p["questions"]:
                choices[p["id"]][q["id"]][name] = select(p, q, density[c], cfg["weight"])
    for k, (model, _) in models.items():
        labels = model.labels_ if model is not None else np.zeros(len(fit), dtype=int)
        counts = np.bincount(labels, minlength=k)
        sizes[str(k)] = {"minimum": int(counts.min()), "median": float(np.median(counts)), "maximum": int(counts.max()), "clusters_lt5": int(sum(counts<5))}
    frozen = {"parameters": selection, "choices": choices, "heldout_papers": splits["heldout"], "cluster_sizes": sizes}
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    # Heldout annotations are first consulted here, after frozen choices are saved.
    records = metrics(test, choices, source)
    (OUT / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"fit_papers": len(fit), "validation_papers": len(val), "heldout_papers": len(test),
               "heldout_questions": sum(len(p["questions"]) for p in test), "selected": {"global": global_choice, "fingerprint": fp_choice},
               "arms": {}, "paired": {}, "cluster_sizes": sizes}
    for arm in ("bm25", "global", "fingerprint"):
        allrows = [q["arms"][arm] for p in records for q in p["records"]]
        vals = [v for v in allrows if v["recall"] is not None]
        summary["arms"][arm] = {"eligible": len(vals), "recall": recall(records, arm),
            "any_hit": sum(v["any"] for v in vals), "all_hit": sum(v["all"] for v in vals), "chars": sum(v["chars"] for v in allrows)}
    for arm, ref in (("fingerprint", "global"), ("fingerprint", "bm25"), ("global", "bm25")):
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sample = [records[rng.randrange(len(records))] for _ in records]
            boot.append(recall(sample, arm) - recall(sample, ref))
        boot.sort()
        summary["paired"][arm+"_minus_"+ref] = {"difference": recall(records, arm)-recall(records, ref), "paper_ci95": [boot[250], boot[9750]]}
    summary["fingerprint_gate"] = summary["paired"]["fingerprint_minus_global"]["paper_ci95"][0] > 0 and summary["arms"]["fingerprint"]["any_hit"] >= summary["arms"]["global"]["any_hit"]
    summary["total_pipeline_sec"] = time.perf_counter() - start
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
