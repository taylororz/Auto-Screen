#!/usr/bin/env python3
"""E5d/E5e: Replay SoM predictions on saved annotated screenshots and element lists."""
import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OLD_COPY = Path("/Users/taylorzhuang/Masterarbeit/Auto-Screen")
LOG = ROOT_DIR / "Qwen_trained" / "diff_results" / "interaction_log.jsonl"
REVIEWED = ROOT_DIR / "Qwen_trained" / "diff_results" / "finetune_dataset.jsonl"
DAY = "2026-09-28"


def load_steps():
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("timestamp", "").startswith(DAY)]
    reviewed = {(r["timestamp"], r["step_name"]): r for r in
                (json.loads(l) for l in open(REVIEWED, encoding="utf-8") if l.strip())
                if r.get("timestamp", "").startswith(DAY)}
    steps = []
    for r in rows:
        img = Path(r["annotated_image_path"].replace("/Users/taylorzhuang/Auto-Screen/", str(OLD_COPY) + "/"))
        elements_file = img.with_suffix(".json")
        gt = reviewed.get((r["timestamp"], r["step_name"]))
        steps.append({"case": f"{r['timestamp']}|{r['step_name']}", "language": r["language"], "step_name": r["step_name"],
                      "instruction": r["instruction"], "annotated_image": str(img), "elements_file": str(elements_file),
                      "target_2809": int(r["target_id"]) if str(r.get("target_id", "")).isdigit() else r.get("target_id"),
                      "ground_truth_id": gt.get("ground_truth_id") if gt else None})
    return steps


def cmd_run(a):
    spec = importlib.util.spec_from_file_location("visual_agent", ROOT_DIR / "Qwen_trained" / "visual_agent.py")
    va = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(va)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8") if l.strip()}
    steps = load_steps()
    print(f"{len(steps)} steps, {len(done)} already completed")
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(steps, 1):
            if s["case"] in done:
                continue
            elements = json.loads(Path(s["elements_file"]).read_text(encoding="utf-8"))
            elements = elements if isinstance(elements, list) else elements["elements"]
            res = va.ask_ai_for_element_id(s["annotated_image"], s["instruction"], elements, language=s["language"])
            row = dict(s, target_replay=res.get("target_id"), status=res.get("status"), raw=str(res.get("rationale"))[:200],
                       time_ms=round(res.get("time_ms", 0)), model_path=va.MODEL_PATH)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(steps)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done:", out)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    same = [r for r in rows if r["target_replay"] == r["target_2809"]]
    with_gt = [r for r in rows if r["ground_truth_id"] is not None]
    correct = [r for r in with_gt if r["target_replay"] == r["ground_truth_id"]]
    diff = Counter(r["step_name"] for r in rows if r["target_replay"] != r["target_2809"])
    by_lang = defaultdict(lambda: [0, 0])
    for r in rows:
        by_lang[r["language"]][0] += r["target_replay"] == r["target_2809"]
        by_lang[r["language"]][1] += 1
    times = sorted(r["time_ms"] for r in rows)
    res = {"steps": len(rows), "same_as_2809": f"{len(same)}/{len(rows)}",
           "correct_vs_reviewed": f"{len(correct)}/{len(with_gt)}",
           "fallback_or_error": sum(1 for r in rows if r["status"] != "analyzed_local"),
           "different_by_step": dict(diff.most_common()),
           "same_by_language": {k: f"{v[0]}/{v[1]}" for k, v in by_lang.items()},
           "median_time_ms": times[len(times) // 2] if times else None}
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
