import os
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pilot_2b_3k.json"), encoding="utf-8"))
print("summary:", json.dumps(d["summary"], ensure_ascii=False))
recs = d["records"]
print("n records:", len(recs))
r0 = recs[0]
print("record keys:", list(r0.keys()))
for k, v in r0.items():
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print(f"  {k}: list[{len(v)}], entry keys: {list(v[0].keys())}")
        if "thinking" in v[0]:
            print(f"    trace lens: {[len(e.get('thinking','')) for e in v]}")
            print(f"    answers: {[e.get('answers') for e in v]}")
    elif isinstance(v, dict):
        print(f"  {k}: dict, keys: {list(v.keys())}")
