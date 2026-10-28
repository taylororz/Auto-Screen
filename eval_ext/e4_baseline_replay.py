#!/usr/bin/env python3
"""
E4: Freigegebene Basis gegenüber fortlaufender Basis (Demonstration, siehe PROTOKOLL.md).

Gibt die abwechselnde CI-Folge A1 -> B5 -> A2 -> B5 mit den lokalen FF3-Läufen wieder und vergleicht
jeden Lauf (a) mit dem vorherigen Lauf und (b) mit der freigegebenen Basis A1.

  python3 eval_ext/e4_baseline_replay.py --out eval_ext/results/e4
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import compare_images  # noqa: E402

FF3 = ROOT_DIR / "ff3"
FOLDERS = ["baseline_DE_DE", "baseline_EN_US", "baseline_ZH_CN", "baseline_ZH_TW"]
PAGES = ["06_vehicle_tab", "07_settings_tab", "01_home_tab", "language_button", "login_field", "13_language_choice"]
SEQUENCE = ["A1", "B5", "A2", "B5"]
BUILD = {"A1": "A", "A2": "A", "B5": "B"}


def find_page(folder, page):
    for name in (f"{page}.png", f"katapp_{page}.png"):
        p = Path(folder) / name
        if p.exists():
            return p
    return None


def intersects(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def compare_runs(base_run, actual_run, out, gt):
    boxes, at_defect = 0, 0
    for folder in FOLDERS:
        for page in PAGES:
            b = find_page(FF3 / "runs" / base_run / "screenshots" / folder, page)
            a = find_page(FF3 / "runs" / actual_run / "screenshots" / folder, page)
            if not b or not a:
                continue
            res = compare_images.compare_ui_screenshots(b, a, out / f"{actual_run}_vs_{base_run}" / folder / f"diff_{page}.png", threshold=38, min_box_size=14)
            rects = [d["bbox"] for d in gt if d["folder"] == folder and d["page"] == page and d.get("bbox") and any(d["bbox"])]
            boxes += len(res["pixel"]["boxes"])
            at_defect += sum(1 for box in res["pixel"]["boxes"] if any(intersects(box, r) for r in rects))
    return {"baseline": base_run, "actual": actual_run, "actual_build": BUILD[actual_run],
            "pixel_boxes": boxes, "pixel_boxes_at_defect_areas": at_defect}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads((FF3 / "seeded_defects.json").read_text(encoding="utf-8"))["defects"]

    result = {"sequence": SEQUENCE, "rolling": [], "approved_A1": []}
    for step in range(1, len(SEQUENCE)):
        result["rolling"].append({"run": step + 1, **compare_runs(SEQUENCE[step - 1], SEQUENCE[step], out, gt)})
        result["approved_A1"].append({"run": step + 1, **compare_runs("A1", SEQUENCE[step], out, gt)})
    for mode in ("rolling", "approved_A1"):
        clean = [r for r in result[mode] if r["actual_build"] == "A"]
        result[f"{mode}_boxes_on_defect_free_builds"] = sum(r["pixel_boxes"] for r in clean)
    (out / "e4_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
