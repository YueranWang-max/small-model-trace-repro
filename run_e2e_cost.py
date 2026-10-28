"""Predeclared, same-corpus lifecycle comparison for the revised manuscript.

prepare freezes questions before ingestion. evaluate uses the independently
generated full-corpus edge DB and never reads gold answers for routing/repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from repro_lib import extract_answer, f1_score, ollama_chat, sample_from_target_chars
from run_distill3 import EXTRACT_SYSTEM, chunk_edges, parse_edges, verify_edge
from run_recall import RESCAN_TASKS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "e2e_128k"
SIZE = 128000
SEED = 11 + SIZE  # same corpus as run_transcribe.py --size 128000
QSEED = 20261001
MODEL = "openbmb/minicpm5-2b"
ENSEMBLE = "qwen3.5:4b"


def graph():
    sample = sample_from_target_chars(random.Random(SEED), SIZE, op_type="bfs")
    edges = set(map(tuple, sample.edges))
    digest = hashlib.sha256("\n".join(f"{u} -> {v}" for u, v in sorted(edges)).encode()).hexdigest()
    return sample, edges, digest


def maps(edges):
    adj, rev = defaultdict(set), defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        rev[v].add(u)
    return adj, rev


def bfs(adj, start, depth):
    visited, frontier = {start}, {start}
    for _ in range(depth):
        frontier = {v for u in frontier for v in adj.get(u, ()) if v not in visited}
        visited.update(frontier)
    return sorted(frontier)


def prepare():
    sample, edges, digest = graph()
    adj, rev = maps(edges)
    rng = random.Random(QSEED)
    parent_nodes = sorted(v for v, ins in rev.items() if 3 <= len(ins) <= 12)
    rng.shuffle(parent_nodes)
    qs = []
    for node in parent_nodes[:6]:
        qs.append({"kind": "parents", "node": node, "depth": None,
                   "question": f"Find the parents of node {node}.",
                   "answer": sorted(rev[node])})
    for depth in (2, 3, 4):
        candidates = sorted(u for u in adj if 4 <= len(bfs(adj, u, depth)) <= 80)
        rng.shuffle(candidates)
        for node in candidates[:2]:
            qs.append({"kind": "bfs", "node": node, "depth": depth,
                       "question": f"Perform a BFS from node {node} with depth {depth}.",
                       "answer": bfs(adj, node, depth)})
    assert len(qs) == 12, len(qs)
    for i, q in enumerate(qs):
        q["id"] = i + 1
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"corpus_size_requested": SIZE, "corpus_chars": sample.graph_chars,
               "corpus_sha256": digest, "graph_seed": SEED, "question_seed": QSEED,
               "n_edges": len(edges), "questions": qs}
    path = OUT / "questions.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == payload
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Frozen {len(qs)} questions before ingestion: {path}; graph={digest}", flush=True)


def call(model, system, user, *, think, temperature, num_predict):
    t = time.perf_counter()
    g = ollama_chat(model, system, user, num_ctx=16384, think=think,
                    temperature=temperature, num_predict=num_predict)
    sec = time.perf_counter() - t
    raw = g.raw
    return g, {"wall_sec": round(sec, 3),
               "ollama_duration_sec": round(raw.get("total_duration", 0) / 1e9, 3),
               "prompt_tokens": raw.get("prompt_eval_count"),
               "generation_tokens": raw.get("eval_count"),
               "model": model}


def alarm_repair(db, chunks, idx_left, idx_right, q):
    """Existing node-zero-output alarm, with repairs shared across later queries."""
    calls, repaired, unrepaired = [], 0, 0
    n_steps = q["depth"] if q["kind"] == "bfs" else 1
    frontier, visited = {q["node"]}, {q["node"]}
    for _ in range(n_steps):
        for node in sorted(frontier):
            relevant = idx_right if q["kind"] == "parents" else idx_left
            for ci in sorted(relevant.get(node, ())):
                if any((v == node if q["kind"] == "parents" else u == node)
                       and verify_edge(u, v, chunks[ci]) for u, v in db):
                    continue
                fixed = set()
                for name, temp, who, template in RESCAN_TASKS:
                    model = ENSEMBLE if who == "ens" else MODEL
                    task = (f"Copy every edge ending in {node}, one per line."
                            if q["kind"] == "parents" else template.replace("{u}", node))
                    prompt = task + "\n\nPortion of the edge list:\n" + chunks[ci]
                    started = time.perf_counter()
                    try:
                        g, meta = call(model, EXTRACT_SYSTEM, prompt, think=False,
                                       temperature=temp, num_predict=3072)
                    except Exception as exc:
                        g = None
                        meta = {"wall_sec": round(time.perf_counter() - started, 3),
                                "prompt_tokens": None, "generation_tokens": None,
                                "model": model, "error": repr(exc)}
                    meta.update({"chunk": ci, "node": node, "rescan": name})
                    fixed = ({e for e in parse_edges(g.content)
                              if (e[1] == node if q["kind"] == "parents" else e[0] == node)
                              and verify_edge(e[0], e[1], chunks[ci])} if g else set())
                    meta.update({"raw_content": g.content if g else "",
                                 "verified_edges": sorted(fixed)})
                    calls.append(meta)
                    if fixed:
                        db.update(fixed)
                        repaired += 1
                        break
                if not fixed:
                    unrepaired += 1
        if q["kind"] == "parents":
            break
        adj, _ = maps(db)
        frontier = {v for u in frontier for v in adj.get(u, ()) if v not in visited}
        visited.update(frontier)
    return calls, repaired, unrepaired


def evidence(db, q):
    adj, rev = maps(db)
    if q["kind"] == "parents":
        return "Verified edges:\n" + "\n".join(f"{u} -> {q['node']}" for u in sorted(rev[q["node"]]))
    frontier, visited, lines = {q["node"]}, {q["node"]}, []
    for d in range(1, q["depth"] + 1):
        local = sorted((u, v) for u in frontier for v in adj.get(u, ()))
        lines.append(f"Level {d} verified edges:\n" + "\n".join(f"{u} -> {v}" for u, v in local))
        frontier = {v for _, v in local if v not in visited}
        visited.update(frontier)
    return "\n".join(lines)


def read_answer(material, q, budget):
    prompt = ("Use only the provided directed edges to answer the operation.\n"
              + material + "\n\nOperation: " + q["question"]
              + "\nReturn exactly one line: Final Answer: [node1, node2]. Use [] if empty.")
    started = time.perf_counter()
    try:
        g, meta = call(MODEL, "You solve directed-graph algorithm problems.", prompt,
                       think=True, temperature=0.0, num_predict=budget)
    except Exception as exc:
        return {"answer": [], "malformed": True, "f1": 0.0, "em": 0,
                "prompt_chars": len(prompt), "prompt_tokens": None,
                "generation_tokens": None, "raw_content": "", "thinking_chars": 0,
                "wall_sec": round(time.perf_counter() - started, 3),
                "error": repr(exc)}
    # Only the visible answer channel counts; a thought-only result is a failure.
    ans, malformed = extract_answer(g.content, "")
    meta.update({"answer": ans, "malformed": malformed,
                 "f1": f1_score(ans, q["answer"]),
                 "em": int(set(ans) == set(q["answer"])),
                 "prompt_chars": len(prompt), "raw_content": g.content,
                 "thinking_chars": len(g.thinking),
                 "response_sha256": hashlib.sha256(json.dumps(g.raw, sort_keys=True).encode()).hexdigest()})
    return meta


def bm25_index(chunks):
    from run_stageC import BM25
    return BM25(chunks)


def evaluate(budget):
    frozen = json.loads((OUT / "questions.json").read_text(encoding="utf-8"))
    sample, truth, digest = graph()
    assert digest == frozen["corpus_sha256"]
    ingest = json.loads((OUT / "ingest.json").read_text(encoding="utf-8"))
    db = set(map(tuple, json.loads((OUT / "ingest_db.json").read_text(encoding="utf-8"))["db"]))
    assert db <= truth, "Source verification failed"
    chunks = chunk_edges(sample, 4000)
    t_index = time.perf_counter()
    idx_left, idx_right = defaultdict(set), defaultdict(set)
    for ci, ch in enumerate(chunks):
        for u in re.findall(r"^n\d+(?= ->)", ch, re.MULTILINE):
            idx_left[u].add(ci)
        for v in re.findall(r"(?<=-> )n\d+$", ch, re.MULTILINE):
            idx_right[v].add(ci)
    sdr_index_sec = time.perf_counter() - t_index
    t_bm = time.perf_counter()
    bm = bm25_index(chunks)
    bm25_index_sec = time.perf_counter() - t_bm
    t_parse = time.perf_counter()
    parsed_raw = set()
    for ch in chunks:
        parsed_raw.update(re.findall(r"^(n\d+) -> (n\d+)$", ch, re.MULTILINE))
    parser_build_sec = time.perf_counter() - t_parse
    assert parsed_raw == truth
    raw_adj, raw_rev = maps(parsed_raw)
    out_file = OUT / ("results.json" if budget == 4096 else f"results_budget{budget}.json")
    records = []
    if out_file.exists():
        previous = json.loads(out_file.read_text(encoding="utf-8"))
        assert previous["corpus_sha256"] == digest
        records = previous["records"]
        assert [r["id"] for r in records] == [q["id"] for q in frozen["questions"][:len(records)]]
        for r in records:
            db.update(map(tuple, r.get("new_verified_edges", [])))
        print(f"Resuming after {len(records)} completed questions", flush=True)
    for q in frozen["questions"][len(records):]:
        initial = set(db)
        alarm_calls, repaired, unrepaired = alarm_repair(db, chunks, idx_left, idx_right, q)
        ev = evidence(db, q)
        t_ret = time.perf_counter()
        selected = bm.search(q["question"], 4)
        retrieval_sec = time.perf_counter() - t_ret
        retrieved = "Retrieved source chunks:\n" + "\n\n".join(chunks[i] for i in selected)
        if q["id"] % 2:
            state = read_answer(ev, q, budget)
            rag = read_answer(retrieved, q, budget)
            order = "state-first"
        else:
            rag = read_answer(retrieved, q, budget)
            state = read_answer(ev, q, budget)
            order = "bm25-first"
        a0, r0 = maps(initial)
        a1, r1 = maps(db)
        det0 = sorted(r0[q["node"]]) if q["kind"] == "parents" else bfs(a0, q["node"], q["depth"])
        det1 = sorted(r1[q["node"]]) if q["kind"] == "parents" else bfs(a1, q["node"], q["depth"])
        t_cpu = time.perf_counter()
        cpu_ans = sorted(raw_rev[q["node"]]) if q["kind"] == "parents" else bfs(raw_adj, q["node"], q["depth"])
        cpu_sec = time.perf_counter() - t_cpu
        rec = {"id": q["id"], "question": q["question"], "kind": q["kind"],
               "readout_order": order,
               "depth": q["depth"], "gold_size": len(q["answer"]),
               "det_pre_f1": f1_score(det0, q["answer"]),
               "det_post_f1": f1_score(det1, q["answer"]),
               "raw_parser_f1": f1_score(cpu_ans, q["answer"]),
               "raw_parser_query_sec": round(cpu_sec, 6),
               "alarm_calls": alarm_calls, "repaired": repaired, "unrepaired": unrepaired,
               "new_verified_edges": sorted(db - initial),
               "state": state, "bm25_top4": {**rag, "selected_chunks": selected,
                                                "retrieval_sec": round(retrieval_sec, 6)}}
        records.append(rec)
        out = {"corpus_sha256": digest, "questions_sha256": hashlib.sha256((OUT / "questions.json").read_bytes()).hexdigest(),
               "readout_num_predict": budget,
               "ingest": {"wall_sec": ingest["elapsed_sec"], "calls": ingest["transcribe_calls"],
                          "recall": ingest["recall"], "sdr_index_sec": sdr_index_sec,
                          "bm25_index_sec": bm25_index_sec, "raw_parser_build_sec": parser_build_sec},
               "records": records}
        out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"q{q['id']}/{len(frozen['questions'])} state={state['f1']:.3f} bm25={rag['f1']:.3f} alarm={len(alarm_calls)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "evaluate"])
    parser.add_argument("--num-predict", type=int, default=4096)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        evaluate(args.num_predict)
