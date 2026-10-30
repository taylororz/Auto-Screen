#!/usr/bin/env python3
"""E3b: Sub-image refinement on verified Direct Grounding flow steps (KatApp, Bluesky)."""
import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "Qwen_trained" / "diff_results" / "finetune_dataset_DG.jsonl"
TOLERANCE = 40
KATAPP_STEPS = {"home_tab", "triage_dropdown", "triage_option", "scan_tab", "overview_tab", "vehicle_tab", "settings_tab",
                "activation_token_button", "login_field", "login_save_button", "language_button", "language_choice",
                "initial_login_save_button"}


def app_of(row):
    return "KatApp" if row["step_name"] in KATAPP_STEPS else "Bluesky"


def cmd_run(a):
    for key in ("DG_SUBIMAGE_REFINE", "DG_SUBIMAGE_OVERLAP", "DG_SUBIMAGE_MAX_SHIFT"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT_DIR / "Qwen_trained"))
    import cv2
    import visual_agent_DG as dg

    rows = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8") if l.strip()}
        print(f"Resuming: {len(done)} steps already completed")
    tmp = out.parent / "e3b_tmp"
    tmp.mkdir(exist_ok=True)
    counts = defaultdict(int)
    for r in rows:
        counts[app_of(r)] += 1
    print(f"{len(rows)} steps: {dict(counts)}")

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            case = f"{r['timestamp']}|{r['step_name']}"
            if case in done:
                continue
            img_path = tmp / f"step_{i:04d}.png"
            shutil.copyfile(r["screenshot_path"], img_path)
            img = cv2.imread(str(img_path))
            cx, cy = int(r["center_x"]), int(r["center_y"])
            start = time.time()
            ref = dg._refine_in_subimage(img, img_path, r["instruction"], cx, cy)
            row = {"case": case, "index": i, "app": app_of(r), "language": r["language"], "step_name": r["step_name"],
                   "instruction": r["instruction"], "screenshot_path": r["screenshot_path"], "confirmed_point": [cx, cy],
                   "accepted": ref["accepted"], "refined_point": ref.get("refined_point"), "final_point": ref.get("point", [cx, cy]),
                   "reason": ref.get("reason"), "subimage": ref["subimage"], "time_refine_ms": round((time.time() - start) * 1000)}
            fx, fy = row["final_point"]
            row["shift"] = round(math.hypot(fx - cx, fy - cy), 1)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            for p in tmp.glob(f"step_{i:04d}*"):
                p.unlink()
            if i % 20 == 0:
                print(f"{i}/{len(rows)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done:", out)


def cmd_review(a):
    """Export review images for accepted refinements with shift > 40 px."""
    import cv2
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if r["accepted"] and r["shift"] > TOLERANCE]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E3b: Manual Review", "",
             "Red = verified initial point, blue = refined point.",
             "Set true or false for each case in manual.json.", ""]
    manual = {}
    for r in todo:
        img = cv2.imread(r["screenshot_path"])
        (cx, cy), (fx, fy) = r["confirmed_point"], r["final_point"]
        for (x, y), color in (((cx, cy), (0, 0, 230)), ((fx, fy), (230, 90, 0))):
            cv2.circle(img, (x, y), 34, color, 6)
            cv2.line(img, (x - 50, y), (x + 50, y), color, 4)
            cv2.line(img, (x, y - 50), (x, y + 50), color, 4)
        cv2.putText(img, r["instruction"][:70], (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        name = f"case_{r['index']:04d}.png"
        cv2.imwrite(str(out / name), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        lines.append(f"- `{name}` ({r['app']}, {r['language']}, {r['step_name']}, shift {r['shift']} px): {r['instruction']}")
        manual[str(r["index"])] = None
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manual_path = out / "manual.json"
    if not manual_path.exists():
        manual_path.write_text(json.dumps(manual, indent=2), encoding="utf-8")
    print(f"{len(todo)} cases exported for manual review in {out}")


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    manual = json.loads(Path(a.manual).read_text(encoding="utf-8")) if a.manual and Path(a.manual).exists() else {}
    res = {"steps": len(rows)}
    for app in ("KatApp", "Bluesky", "gesamt"):
        R = [r for r in rows if app == "gesamt" or r["app"] == app]
        acc = [r for r in R if r["accepted"]]
        shifts = sorted(r["shift"] for r in acc)
        far = [r for r in acc if r["shift"] > TOLERANCE]
        judged = [manual.get(str(r["index"])) for r in far]
        hits_far = sum(1 for j in judged if j is True)
        times = sorted(r["time_refine_ms"] for r in R)
        res[app] = {
            "steps": len(R), "accepted": len(acc),
            "median_shift_px": shifts[len(shifts) // 2] if shifts else None,
            "accepted_within_40px": sum(1 for s in shifts if s <= TOLERANCE),
            "accepted_over_40px": len(far),
            "over_40px_manual_hit": hits_far, "over_40px_manual_miss": sum(1 for j in judged if j is False),
            "over_40px_not_judged": sum(1 for j in judged if j is None),
            "step_accuracy_with_refinement": f"{len(R) - len(far) + hits_far}/{len(R)}" if all(j is not None for j in judged) else "offen",
            "median_time_refine_ms": times[len(times) // 2] if times else None,
        }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    v = sp.add_parser("review")
    v.add_argument("--run", required=True)
    v.add_argument("--out", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--manual", default=None)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "review": cmd_review, "summary": cmd_summary}[args.cmd](args)
