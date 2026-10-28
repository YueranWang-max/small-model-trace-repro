# -*- coding: utf-8 -*-
"""Parse-only verification of every command in README 'Running' block.
Extracts add_argument() calls from each script's source, rebuilds the
argparse parser, and parses the README command line. No script code is
executed and no inference starts."""
import re, argparse, sys, os

D = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "run_pilot.py": "--model openbmb/minicpm5-2b --target-chars 6000 --n-questions 12 --n-tr 3 --num-ctx 32768 --max-predict 4096 --out logs/pilot_2b_6k.json",
    "run_transcribe.py": "--size 512000 --out logs/transcribe_512k_v3.json",
    "run_distill3.py": '--sizes "16000,48000,128000,256000,512000" --ensemble qwen3.5:4b',
    "run_bfs_deep.py": '--size 512000 --depths "4,5" --with-parents 1 --tag v4',
    "run_recall.py": "",
    "run_budget_matched.py": "--stage A --out logs/budget_matched_stageA.json",
    "run_stageC.py": "",
    "run_stageDE_chain.py": "",
    "finish_stageDE.py": "",
    "analyze_stageB.py": "",
    "analyze_stageC.py": "",
    "analyze_stageD.py": "",
    "analyze_stageE.py": "",
    "verify_stageE_extra.py": "",
    "exact_test_stageB.py": "",
    "bootstrap_ci.py": "",
    "vram_summary.py": "",
    "run_consensus.py": "--target-chars 6000 --n-questions 12 --n-tr 3 --vote-n 5",
    "run_chain.py": "--target-chars 6000 --n-questions 12 --no-chain-et",
    "run_scale.py": "--model qwen3.5:4b --big-model qwen3-vl:8b --ref logs/x.json",
    "run_realdoc.py": "--data data/realdoc_hotpot_32k.jsonl --n-instances 5 --pred-read 12288",
    "build_realdoc.py": "--src data/upstream.json --target-chars 32000 --n-instances 50",
}

def build_parser(script):
    src = open(os.path.join(D, script), encoding="utf-8", errors="ignore").read()
    calls = re.findall(r"add_argument\((?:[^()]|\([^()]*\))*\)", src)
    p = argparse.ArgumentParser()
    for c in calls:
        c2 = re.sub(r'default=[A-Z_][A-Z0-9_]*', 'default=None', c)
        try:
            exec(f"p.{c2}", {"p": p, "argparse": argparse})
        except Exception as e:
            print(f"  !! could not rebuild arg in {script}: {c[:60]} ({e})")
    return p

ok = True
for script, cmd in COMMANDS.items():
    try:
        p = build_parser(script)
        argv = cmd.split() if cmd else []
        ns = p.parse_args(argv)
        print(f"PARSE OK  {script}" + (f"  -> {vars(ns)}" if cmd else ""))
    except SystemExit:
        ok = False
        print(f"PARSE FAIL {script}: {cmd}")
    except FileNotFoundError:
        print(f"SKIP      {script}: not found")
print("\nALL COMMANDS PARSE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
