# -*- coding: utf-8 -*-
"""Ollama 连通性 + thinking 轨迹冒烟测试。结果写 logs/smoke.log (utf-8)。"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
_LOG = open(os.path.join(HERE, "logs", "smoke.log"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    try:
        print(s)
    except Exception:
        pass
    _LOG.write(s + "\n")
    _LOG.flush()


from repro_lib import (  # noqa: E402
    GRAPH_SYSTEM, build_graph_x, extract_answer, ollama_chat,
    sample_from_target_chars,
)

MODELS = ["openbmb/minicpm5-2b", "qwen3.5:4b"]

for model in MODELS:
    p(f"\n{'='*60}\n模型: {model}")
    s = sample_from_target_chars(random.Random(5), 4000, op_type="parents")
    x_head, x_op = build_graph_x(s)
    x = x_head + "\n" + x_op
    for think in (True, False):
        t0 = time.time()
        try:
            g = ollama_chat(model, GRAPH_SYSTEM, x, num_ctx=8192,
                            think=think, temperature=0.7, num_predict=4096,
                            timeout=900)
        except Exception as e:
            p(f"  think={think} 调用失败: {e}")
            continue
        dt = time.time() - t0
        ans, bad = extract_answer(g.content, g.thinking)
        p(f"  think={think:>5}  用时 {dt:6.1f}s  thinking长度={len(g.thinking):>5} "
          f"content长度={len(g.content):>4}  抽取答案={ans[:10]} (错={bad})")
        p(f"    ── 参考答案: {sorted(s.answer)}")
        p(f"    ── 完全正确={set(ans) == set(s.answer)}  F1={__import__('repro_lib').f1_score(ans, s.answer):.3f}")
        p(f"    ── content 前 200 字: {g.content[:200]!r}")
        if g.thinking:
            p(f"    ── thinking 末 200 字: ...{g.thinking[-200:]!r}")

_LOG.close()
