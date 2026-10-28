"""Describe fitted prototypes using fitting papers only; no retuning."""
import json
import numpy as np
import joblib
from run_qasper_fingerprint import OUT, SOURCE, ZONES, prepare

def main():
    split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    saved = joblib.load(OUT / "model.joblib")
    k = saved["selection"]["fingerprint"]["k"]
    model = saved["models"][k][0]
    papers = [prepare(pid, raw[pid]) for pid in split["fit"]]
    x = np.array([p["features"] for p in papers])
    distances = model.transform(saved["scaler"].transform(x))
    profiles = []
    for c in range(k):
        indices = np.where(model.labels_ == c)[0]
        frequencies = x[indices, :10].mean(axis=0)
        nearest = sorted(indices, key=lambda i: distances[i, c])[:3]
        locations = {}
        for z, name in enumerate(ZONES):
            values = [x[i, 10+z] for i in indices if x[i, 10+z] >= 0]
            if values:
                locations[name] = float(np.mean(values))
        profiles.append({"prototype": c, "fitting_papers": len(indices),
            "mean_role_fraction": {z: float(v) for z, v in zip(ZONES, frequencies)},
            "mean_role_location_when_present": locations,
            "representatives": [{"id": papers[i]["id"], "headings": [s["section_name"] for s in raw[papers[i]["id"]]["full_text"]]} for i in nearest]})
    result = {"k": k, "fit_mean_other_role_fraction": float(x[:,9].mean()), "profiles": profiles}
    (OUT / "profiles.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("prototypes", k, "mean other fraction", result["fit_mean_other_role_fraction"])

if __name__ == "__main__":
    main()
