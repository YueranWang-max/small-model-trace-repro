"""Summarize the exploratory no-thinking readout pilots."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "e2e_128k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    records = data["records"]
    assert len(records) == 12 and [r["id"] for r in records] == list(range(1, 13))
    ingest = json.loads((OUT / "ingest.json").read_text(encoding="utf-8"))
    out = {"n": len(records), "model": data.get("model", "openbmb/minicpm5-2b"), "think": data["think"],
           "num_predict": data["num_predict"],
           "ingest_wall_sec_reused": ingest["elapsed_sec"]}
    for arm in ("state", "bm25_top4"):
        rows = [r[arm] for r in records]
        out[arm + "_mean_f1"] = statistics.mean(r["f1"] for r in rows)
        out[arm + "_em"] = statistics.mean(r["em"] for r in rows)
        out[arm + "_readout_wall_sec"] = sum(r["wall_sec"] for r in rows)
        out[arm + "_generation_tokens"] = sum(r.get("generation_tokens") or 0 for r in rows)
        out[arm + "_malformed_count"] = sum(bool(r["malformed"]) for r in rows)
    # The original query alarm/index times are not replayed in this reader-only pilot.
    out["state_lifecycle_lower_bound_sec"] = out["ingest_wall_sec_reused"] + out["state_readout_wall_sec"]
    target = args.path.with_name(args.path.stem.replace("results", "summary") + ".json")
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
