#!/usr/bin/env python3
"""E8: Evaluate SoM predictions on E5d steps with spatial location hints removed from instructions."""
import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REWRITES = {
    "go to the back icon in the top left corner.": "go to the back icon.",
    "click the menu icon in the top left corner": "click the menu icon",
    "Click the Settings gear icon (4th tab from left) in the bottom navigation bar": "Click the Settings gear icon",
    "Enter activation token in the text input field in the center": "Enter activation token in the text input field",
    "Click the Save button below the input field": "Click the Save button",
    "Click the Home icon (1st tab from left) in the bottom navigation bar": "Click the Home icon",
    "Click the triage location dropdown menu in the upper-middle area": "Click the triage location dropdown menu",
    "Click the Scan icon (2nd tab from left) in the bottom navigation bar": "Click the Scan icon",
    "Click the Triages overview icon (3rd tab in the center) in the bottom navigation bar": "Click the Triages overview icon",
    "Click the Vehicle truck icon (5th tab on the far right) in the bottom navigation bar": "Click the Vehicle truck icon",
    "go to the home icon in the bottom bar.": "go to the home icon.",
    "go to # page in the top right corner.": "go to # page.",
    "click the 'Search' icon in the bottom bar": "click the 'Search' icon",
    "click the chat bubble icon in the bottom bar": "click the chat bubble icon",
    "click the 'Notification' icon in the bottom bar": "click the 'Notification' icon",
    "click the '@' icon in the bottom bar": "click the '@' icon",
    "click the home icon in the bottom bar": "click the home icon",
    "Select the Incident option labeled '事件' (1st option in the dropdown list)": "Select the Incident option labeled '事件'",
    "Select the Incident option labeled 'Schadensort' (1st option in the dropdown list)":
        "Select the Incident option labeled 'Schadensort'",
    "Select the Incident option labeled 'Incident' (1st option in the dropdown list)":
        "Select the Incident option labeled 'Incident'",
    "Click the input field under header 'Password'": "Click the input field for 'Password'",
}


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval_ext" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_run(a):
    for key in ("LOCAL_MAX_IMAGE_EDGE", "LOCAL_MODEL_PATH"):
        os.environ.pop(key, None)
    e5d = load("e5d_som_replay")
    spec = importlib.util.spec_from_file_location("visual_agent", ROOT / "Qwen_trained" / "visual_agent.py")
    va = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(va)
    steps = [s for s in e5d.load_steps() if s["instruction"] in REWRITES]
    assert len({s["instruction"] for s in steps}) == len(REWRITES), "not all instructions found"
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8") if l.strip()}
    print(f"{len(steps)} steps with spatial hint, {len(done)} already completed, model {va.MODEL_PATH}", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(steps, 1):
            if s["case"] in done:
                continue
            elements = json.loads(Path(s["elements_file"]).read_text(encoding="utf-8"))
            elements = elements if isinstance(elements, list) else elements["elements"]
            new = REWRITES[s["instruction"]]
            res = va.ask_ai_for_element_id(s["annotated_image"], new, elements, language=s["language"])
            row = dict(s, instruction_original=s["instruction"], instruction=new, target_replay=res.get("target_id"),
                       status=res.get("status"), raw=str(res.get("rationale"))[:200],
                       time_ms=round(res.get("time_ms", 0)), model_path=va.MODEL_PATH)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(steps)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done:", out)


def cmd_summary(a):
    cmp = load("e5e_compare")
    e8 = {r["case"]: r for r in (json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip())}
    e5d = {r["case"]: r for r in (json.loads(l) for l in open(a.e5d, encoding="utf-8") if l.strip())}
    cases = [c for c in e8 if c in e5d and e5d[c]["ground_truth_id"] is not None]
    ok8 = {c: e8[c]["target_replay"] == e8[c]["ground_truth_id"] for c in cases}
    ok5 = {c: e5d[c]["target_replay"] == e5d[c]["ground_truth_id"] for c in cases}
    only5 = sum(ok5[c] and not ok8[c] for c in cases)
    only8 = sum(ok8[c] and not ok5[c] for c in cases)
    res = {"steps_with_hint": len(cases), "correct_without_hint": sum(ok8.values()), "correct_e5d": sum(ok5.values()),
           "only_e5d_correct": only5, "only_e8_correct": only8,
           "mcnemar_exact_p": round(cmp.mcnemar_exact(only5, only8), 6),
           "fallback_or_error": sum(1 for c in cases if e8[c]["status"] != "analyzed_local")}
    by = defaultdict(lambda: [0, 0])
    for c in cases:
        key = f"{cmp.app_of(e8[c])} | {e8[c]['instruction_original']}"
        by[key][0] += ok8[c]
        by[key][1] += 1
    res["by_instruction"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())}
    # all 598 steps: E8 where the hint was removed, E5d otherwise
    rows = [dict(e8.get(c, r), target_replay=(e8[c] if c in e8 else r)["target_replay"]) for c, r in e5d.items()
            if r["ground_truth_id"] is not None]
    flows = defaultdict(lambda: [0, 0])
    for (app, lang), items in cmp.flows_of(rows):
        wrong = any(r["target_replay"] != r["ground_truth_id"] for r in items)
        flows[app][0] += not wrong
        flows[app][1] += 1
    res["flows_without_error"] = {k: f"{v[0]}/{v[1]}" for k, v in flows.items()}
    times = sorted(e8[c]["time_ms"] for c in cases)
    res["median_time_ms"] = times[len(times) // 2] if times else None
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True)
    s = sub.add_parser("summary"); s.add_argument("--run", required=True); s.add_argument("--e5d", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    main()
