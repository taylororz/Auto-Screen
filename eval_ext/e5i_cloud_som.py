#!/usr/bin/env python3
"""E5i: Evaluate cloud models on A0 SoM screenshots at original and 1600 px resolution."""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

spec = importlib.util.spec_from_file_location("e5h", ROOT_DIR / "eval_ext" / "e5h_renumber.py")
e5h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5h)

MODELS = [("openai", "gpt-6-sol", "GPT-6 Sol"), ("openai", "gpt-6-luna", "GPT-6 Luna"),
          ("anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5")]
SIZES = [("Original", "0"), ("max. 1600 px", None)]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def load_keys():
    for line in (ROOT_DIR / ".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() in KEYS and value.strip():
            os.environ[name.strip()] = value.strip().strip('"').strip("'")


def cmd_run(a):
    load_keys()
    from common_agent.agent_facade import make_agent_facade
    out_dir = ROOT_DIR / "eval_ext" / "results" / "e5i"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for provider, model_name, label in MODELS:
            agent = make_agent_facade(provider=provider, mode="som", model_name=model_name, model_dir=out_dir)
            for size, env in SIZES:
                if env is None:
                    os.environ.pop("MAX_IMAGE_EDGE", None)
                else:
                    os.environ["MAX_IMAGE_EDGE"] = env
                for lang, stem in e5h.CASES.items():
                    meta = json.loads((e5h.STATES / f"{stem}_som.json").read_text(encoding="utf-8"))
                    elements = meta["elements"]
                    boxes = [(e["bbox"]["x"], e["bbox"]["y"], e["bbox"]["width"], e["bbox"]["height"]) for e in elements]
                    img = e5h.IMG_DIR / f"{stem.split('/')[0][9:]}_A0.png"
                    res = agent.ask_ai_for_element_id(str(img), e5h.INSTRUCTION, elements, language=lang)
                    tid = res.get("target_id")
                    role = e5h.roles(boxes).get(tid - 1, "andere") if isinstance(tid, int) and 1 <= tid <= len(boxes) else "ungueltig"
                    row = {"model": label, "model_name": model_name, "size": size, "language": lang, "target_id": tid,
                           "chosen_role": role, "status": res.get("status"), "confidence": res.get("confidence"),
                           "rationale": str(res.get("rationale"))[:200], "reason": res.get("reason"),
                           "time_ms": round(res.get("time_ms", 0)), "image": str(img)}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(label, size, lang, "->", tid, role, row["status"], flush=True)
    print("done:", a.out)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    res = defaultdict(dict)
    for r in rows:
        res[f"{r['model']} | {r['size']}"][r["language"]] = f"{r['chosen_role']} ({r['target_id']}, {r['status']})"
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[args.cmd](args)
