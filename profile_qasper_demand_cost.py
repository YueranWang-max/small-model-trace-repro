"""Cost probe for the fixed global prior; no hyperparameter search or retuning."""
import json
import time
import numpy as np
import joblib
from run_qasper_prefix_vs_bm25 import ROOT
from run_qasper_fingerprint import prepare, select, fit_prior
from run_qasper_demand_transfer import OUT, SOURCE, MODEL_PATH

def main():
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "logs/qasper_fingerprint/split.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "data/qasper_official/qasper-train-v0.3.json").read_text(encoding="utf-8"))
    test = json.loads(SOURCE.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    saved = joblib.load(MODEL_PATH)
    load_sec = time.perf_counter() - t0
    density = saved["models"][1][1][0]
    t0 = time.perf_counter()
    fitting = [prepare(pid, train[pid]) for pid in split["fit"]]
    train_prepare_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    rebuilt = fit_prior(fitting, np.zeros(len(fitting), dtype=int), train)[0]
    train_density_sec = time.perf_counter() - t0
    assert np.allclose(rebuilt, density, rtol=0, atol=1e-12)
    t0 = time.perf_counter()
    cohort = [prepare(pid, test[pid]) for pid in frozen["reader_paper_ids"]]
    reader_prepare_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for p in cohort:
        for q in p["questions"]:
            assert q["rank"][:8] == frozen["choices"][p["id"]][q["id"]]["bm25"]
            assert select(p, q, density, 0.5) == frozen["choices"][p["id"]][q["id"]]["global"]
    prior_selection_sec = time.perf_counter() - t0
    result = {"train_prepare_sec": train_prepare_sec, "train_density_sec": train_density_sec,
        "cold_fixed_model_setup_sec": train_prepare_sec + train_density_sec,
        "model_load_sec": load_sec, "reader_cohort_common_prepare_sec": reader_prepare_sec,
        "reader_cohort_prior_selection_sec": prior_selection_sec,
        "rebuilt_density_matches": True, "selections_match": True,
        "caveat": "A single CPU prototype probe; common preparation includes unused fingerprints and rankings. Fixed-model setup excludes earlier method search; not causal serving-speed/energy attribution."}
    summary_path = OUT / "reader_summary.json"
    if summary_path.exists():
        reader = json.loads(summary_path.read_text(encoding="utf-8"))
        common = reader_prepare_sec
        result["warm_lifecycle_wall_sec"] = {"bm25": reader["arms"]["bm25"]["wall_sec"] + common,
            "global": reader["arms"]["global"]["wall_sec"] + common + prior_selection_sec + load_sec}
        result["cold_lifecycle_wall_sec"] = {"bm25": result["warm_lifecycle_wall_sec"]["bm25"],
            "global": result["warm_lifecycle_wall_sec"]["global"] + result["cold_fixed_model_setup_sec"]}
    (OUT / "cost_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
