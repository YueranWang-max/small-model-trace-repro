"""Losslessly factor cached answers into quote byte pointers plus literals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analyze_qasper_pointer_oracle import serialize_paper, vbytes

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
IN = ROOT / "logs/qasper_known_question_span_v2/results.json"
OUT = ROOT / "logs/qasper_known_question_span_v2/factorization.json"


def factor(answer, quote, source):
    chars = list(answer)
    n = len(chars)
    quote_bytes = quote.encode("utf-8") if quote else b""
    quote_at = source.find(quote_bytes) if quote_bytes else -1
    dp = [None] * (n+1)
    dp[n] = (0, [])
    for i in range(n-1, -1, -1):
        best = None
        for j in range(i+1, n+1):
            fragment = "".join(chars[i:j]).encode("utf-8")
            options = [(1+vbytes(len(fragment))+len(fragment), {"op":"LITERAL", "text":fragment.decode("utf-8")})]
            if quote_at >= 0:
                local = quote_bytes.find(fragment)
                if local >= 0:
                    start = quote_at + local
                    options.append((1+vbytes(start)+vbytes(len(fragment)),
                                    {"op":"COPY", "start":start, "length":len(fragment)}))
            for cost, part in options:
                candidate = (cost + dp[j][0], [part] + dp[j][1])
                if best is None or (candidate[0],len(candidate[1])) < (best[0],len(best[1])):
                    best = candidate
        dp[i] = best
    pieces = dp[0][1]
    decoded = b"".join(source[p["start"]:p["start"]+p["length"]] if p["op"]=="COPY"
                       else p["text"].encode("utf-8") for p in pieces).decode("utf-8")
    assert decoded == answer
    return pieces, vbytes(len(pieces)) + dp[0][0]


def main():
    data = json.loads(DEV.read_text(encoding="utf-8"))
    run = json.loads(IN.read_text(encoding="utf-8"))
    rows = []
    for paper in run["papers"]:
        source = serialize_paper(data[paper["paper_id"]])
        assert paper["source_sha256"] == hashlib.sha256(source).hexdigest()
        for q in paper["records"]:
            arm = q["arms"]["compiler"]
            answer = arm["answer"]
            quote = arm["quote"] if arm["quote_valid"] else ""
            pieces, encoded_bytes = factor(answer, quote, source)
            answer_bytes = len(answer.encode("utf-8"))
            direct_bytes = 1 + vbytes(answer_bytes) + answer_bytes
            rows.append({"paper_id":paper["paper_id"], "id":q["id"],
                         "valid_quote": bool(quote), "answer":answer,
                         "direct_bytes":direct_bytes, "factor_bytes":encoded_bytes,
                         "pieces":pieces})
    summary = {"status":"post_hoc_lossless_encoding_of_archived_model_answers_not_QA_accuracy",
               "n_questions":len(rows), "direct_payload_bytes":sum(x["direct_bytes"] for x in rows),
               "factor_payload_bytes":sum(x["factor_bytes"] for x in rows),
               "bytes_saved":sum(x["direct_bytes"]-x["factor_bytes"] for x in rows),
               "valid_quote":sum(x["valid_quote"] for x in rows),
               "any_copy":sum(any(p["op"]=="COPY" for p in x["pieces"]) for x in rows),
               "mean_pieces":sum(len(x["pieces"]) for x in rows)/len(rows),
               "max_pieces":max(len(x["pieces"]) for x in rows),
               "note":"Payload only; source, question keys, hash, evidence pointer and codebook are additional costs."}
    OUT.write_text(json.dumps({"summary":summary,"records":rows},indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
