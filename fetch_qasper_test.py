"""Fetch official Qasper test data without extracting or executing archive code."""
import io
import json
import tarfile
import urllib.request
from run_qasper_prefix_vs_bm25 import ROOT, sha

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
OUT = ROOT / "data/qasper_official"

def main():
    target = OUT / "qasper-test-v0.3.json"
    manifest = OUT / "test_provenance.json"
    if not target.exists():
        payload = urllib.request.urlopen(URL, timeout=60).read()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.rsplit("/",1)[-1] == target.name]
            assert len(members) == 1
            raw = archive.extractfile(members[0]).read()
            data = json.loads(raw)
            assert isinstance(data, dict) and len(data) > 100
            target.write_bytes(raw)
        manifest.write_text(json.dumps({"url": URL, "archive_sha256": sha(payload),
            "json_sha256": sha(raw), "license": "CC BY 4.0", "papers": len(data),
            "attribution": "Dasigi et al. 2021, Qasper, AllenAI; https://aclanthology.org/2021.naacl-main.365/",
            "loader_source": "https://huggingface.co/datasets/allenai/qasper/raw/main/qasper.py"}, indent=2), encoding="utf-8")
    data = json.loads(target.read_text(encoding="utf-8"))
    for name in ("qasper-train-v0.3.json", "qasper-dev-v0.3.json"):
        previous = json.loads((OUT / name).read_text(encoding="utf-8"))
        assert not set(data) & set(previous)
    print("test papers", len(data), "questions", sum(len(p["qas"]) for p in data.values()), "sha256", sha(target.read_bytes()))

if __name__ == "__main__":
    main()
