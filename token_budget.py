# -*- coding: utf-8 -*-
"""
token_budget —— Ollama 无 tokenizer 端点, 用 eval_count 实测校准 chars/token。
校准常数缓存 logs/token_calib.json; 误差随每次调用滚动更新(指数平均)。
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(BASE, "logs", "token_calib.json")
DEFAULT_CPT = 3.4  # 英文文本的兜底 chars/token


def _load():
    try:
        return json.load(open(CALIB, encoding="utf-8"))
    except Exception:
        return {}


def _save(d):
    os.makedirs(os.path.dirname(CALIB), exist_ok=True)
    json.dump(d, open(CALIB, "w", encoding="utf-8"), indent=2)


def update_from_response(model: str, prompt_chars: int, raw: dict):
    """每次调用后用 prompt_eval_count 滚动校准 chars/token。"""
    pec = raw.get("prompt_eval_count") or 0
    if not pec or not prompt_chars:
        return
    cpt = prompt_chars / pec
    if not (1.5 < cpt < 8):  # 异常值防护
        return
    d = _load()
    old = d.get(model, {}).get("cpt")
    if old:
        d[model]["cpt"] = round(0.7 * old + 0.3 * cpt, 4)
        d[model]["n"] = d[model].get("n", 0) + 1
    else:
        d[model] = {"cpt": round(cpt, 4), "n": 1}
    _save(d)


def chars_per_token(model: str) -> float:
    return _load().get(model, {}).get("cpt", DEFAULT_CPT)


def budget_chars(model: str, budget_tokens: int) -> int:
    return int(budget_tokens * chars_per_token(model))


def est_tokens(model: str, text: str) -> int:
    return int(len(text) / chars_per_token(model))
