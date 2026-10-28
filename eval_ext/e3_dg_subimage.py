#!/usr/bin/env python3
"""
E3: Teilbild-Verfeinerung bei Direct Grounding auf dem bereinigten iOS-1K-Split (siehe PROTOKOLL.md).

Je Fall wird das feinabgestimmte Modell einmal ohne Verfeinerung aufgerufen (a). Seine Vorhersage
ist der erste Schritt der Verfeinerung (b). Beide Punkte werden mit dem Mittelpunkt des Zielelements
verglichen. Der Lauf ist fortsetzbar.

  python3 eval_ext/e3_dg_subimage.py run --n 500 --out eval_ext/results/e3.jsonl
  python3 eval_ext/e3_dg_subimage.py summary --run eval_ext/results/e3.jsonl --out eval_ext/results/e3_summary.json
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT_DIR / "datasets" / "dataset_ios1k" / "ios1k_som_eval.jsonl"
TRAIN_FILE = ROOT_DIR / "qwen_vl_dataset_combined.json"
TOLERANCES = (40, 27)


def stem_id(path):
    """'.../0783_som.png' oder '.../0783.png' -> '0783' (wie ios1k_som_eval.py)"""
    s = Path(path).stem
    return s[:-4] if s.endswith("_som") else s


def clean_cases(n, seed=42):
    """Bereinigter Split wie in ios1k_som_eval.py (Bilder nicht im Training), Stichprobe mit Seed 42."""
    train_ids = {stem_id(x["images"][0]) for x in json.load(open(TRAIN_FILE, encoding="utf-8"))}
    recs = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8") if l.strip()]
    clean = [r for r in recs if stem_id(r["screenshot_path"]) not in train_ids and not r.get("discarded", False)]
    print(f"bereinigt: {len(clean)} Fälle auf {len({stem_id(r['screenshot_path']) for r in clean})} Bildern")
    if n and n < len(clean):
        clean = random.Random(seed).sample(clean, n)
        print(f"Stichprobe: {len(clean)} Fälle (Seed {seed})")
    return clean


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cmd_run(a):
    for key in ("DG_SUBIMAGE_REFINE", "DG_SUBIMAGE_OVERLAP", "DG_SUBIMAGE_MAX_SHIFT"):
        os.environ.pop(key, None)  # Standardwerte aus 2f50706, Verfeinerung wird direkt aufgerufen
    sys.path.insert(0, str(ROOT_DIR / "Qwen_trained"))
    import cv2
    import visual_agent_DG as dg

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8") if l.strip()}
        print(f"Fortsetzung: {len(done)} Fälle bereits erledigt")
    tmp = out.parent / "e3_tmp"
    tmp.mkdir(exist_ok=True)

    cases = clean_cases(a.n)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(cases, 1):
            if r["timestamp"] in done:
                continue
            src = Path(r["screenshot_path"]).expanduser()
            img_path = tmp / f"{stem_id(src)}.png"
            shutil.copyfile(src, img_path)
            gx, gy = float(r["gt_x"]), float(r["gt_y"])

            (ax, ay), decision = dg.predict_target_coordinates(img_path, r["instruction"])
            row = {"case": r["timestamp"], "image": stem_id(src), "instruction": r["instruction"],
                   "element_type": r["step_name"].rsplit("_", 1)[-1], "gt": [gx, gy],
                   "status_a": decision.get("status"), "point_a": [ax, ay], "time_a_ms": round(decision.get("time_ms", 0))}
            bx, by = ax, ay
            refinement = None
            if decision.get("status") == "analyzed_direct_grounding":
                start = time.time()
                img = cv2.imread(str(img_path))
                refinement = dg._refine_in_subimage(img, img_path, r["instruction"], ax, ay)
                row["time_refine_ms"] = round((time.time() - start) * 1000)
                if refinement["accepted"]:
                    bx, by = refinement["point"]
            row["point_b"] = [bx, by]
            row["refinement"] = {k: v for k, v in (refinement or {}).items() if k != "raw_output"} or None
            row["dist_a"] = round(math.hypot(ax - gx, ay - gy), 1)
            row["dist_b"] = round(math.hypot(bx - gx, by - gy), 1)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            for p in tmp.glob(f"{stem_id(src)}*"):
                p.unlink()
            if i % 10 == 0:
                print(f"{i}/{len(cases)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("fertig:", out)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    n = len(rows)
    res = {"cases": n, "images": len({r["image"] for r in rows}),
           "fallback_or_error_a": sum(r["status_a"] != "analyzed_direct_grounding" for r in rows)}
    refined = [r for r in rows if r["refinement"]]
    res["refinement_attempted"] = len(refined)
    res["refinement_accepted"] = sum(r["refinement"]["accepted"] for r in refined)
    for tol in TOLERANCES:
        ha = [r["dist_a"] <= tol for r in rows]
        hb = [r["dist_b"] <= tol for r in rows]
        only_a = sum(x and not y for x, y in zip(ha, hb))
        only_b = sum(y and not x for x, y in zip(ha, hb))
        res[f"tol_{tol}px"] = {
            "hits_without": sum(ha), "hits_with": sum(hb),
            "acc_without": round(sum(ha) / n, 4), "acc_without_ci95": wilson(sum(ha), n),
            "acc_with": round(sum(hb) / n, 4), "acc_with_ci95": wilson(sum(hb), n),
            "only_without_hit": only_a, "only_with_hit": only_b, "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
        }
        by = defaultdict(lambda: [0, 0, 0])
        for r, x, y in zip(rows, ha, hb):
            by[r["element_type"]][0] += x
            by[r["element_type"]][1] += y
            by[r["element_type"]][2] += 1
        res[f"tol_{tol}px"]["by_element_type"] = {k: f"ohne {x}/{m} ({x/m:.1%}), mit {y}/{m} ({y/m:.1%})" for k, (x, y, m) in sorted(by.items())}
    da = sorted(r["dist_a"] for r in rows)
    db = sorted(r["dist_b"] for r in rows)
    res["median_dist_without"] = da[n // 2]
    res["median_dist_with"] = db[n // 2]
    ta = sorted(r["time_a_ms"] for r in rows)
    tr = sorted(r.get("time_refine_ms", 0) for r in refined)
    res["median_time_without_ms"] = ta[n // 2]
    res["median_time_refine_ms"] = tr[len(tr) // 2] if tr else None
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--out", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_summary(args)
