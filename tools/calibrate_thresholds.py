#!/usr/bin/env python3
"""Grid-search diff threshold and minimum box size on annotated screenshot pairs."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import compare_images  # noqa: E402

DEFAULT_FOLDERS = ["baseline_DE_DE", "baseline_EN_US", "baseline_ZH_CN", "baseline_ZH_TW"]


def _find_page(folder, page):
    for name in (f"{page}.png", f"katapp_{page}.png"):
        path = Path(folder) / name
        if path.exists():
            return path
    return None


def _intersects(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _load_pairs(sets, folders, pages):
    """Collects (baseline, actual, defects) per folder and page of all sets."""
    pairs, defects_total, skipped = [], [], []
    for set_index, (baseline_root, actual_root, gt_file) in enumerate(sets, start=1):
        gt = [] if gt_file.lower() == "none" else json.loads(Path(gt_file).read_text(encoding="utf-8"))["defects"]
        gt_by_page = defaultdict(list)
        for defect in gt:
            if not defect.get("bbox"):
                continue
            gt_by_page[(defect["folder"], defect["page"])].append({**defect, "set": set_index})
        defects_total += [f"{set_index}:{d['id']}" for d in gt]
        for folder in folders:
            for page in pages:
                baseline = _find_page(Path(baseline_root) / folder, page)
                actual = _find_page(Path(actual_root) / folder, page)
                if not baseline or not actual:
                    skipped.append((set_index, folder, page))
                    continue
                pairs.append({"set": set_index, "folder": folder, "page": page, "baseline": baseline,
                              "actual": actual, "defects": gt_by_page.get((folder, page), [])})
    return pairs, defects_total, skipped


def main():
    ap = argparse.ArgumentParser(description="Calibrate threshold and minimum box size on annotated screenshot pairs")
    ap.add_argument("--set", nargs=3, action="append", required=True, metavar=("BASELINE_ROOT", "ACTUAL_ROOT", "GT_JSON_OR_NONE"))
    ap.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS)
    ap.add_argument("--pages", nargs="+", required=True, help="Page names (file names without .png)")
    ap.add_argument("--thresholds", nargs="+", type=int, default=[20, 26, 30, 34, 38, 42, 46, 50])
    ap.add_argument("--min-box-sizes", nargs="+", type=int, default=[10, 12, 14, 16, 20])
    ap.add_argument("--mask-top", type=int, default=140)
    ap.add_argument("--mask-bottom", type=int, default=70)
    ap.add_argument("--max-clean-fp", type=int, default=0, help="Allowed boxes on pages without a defect for the recommendation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, defects_total, skipped = _load_pairs(args.set, args.folders, args.pages)
    if not pairs:
        ap.error("No screenshot pairs found")
    print(f"{len(pairs)} pairs, {len(defects_total)} annotated defects, {len(skipped)} pages missing")

    rows = []
    for threshold in args.thresholds:
        for min_box in args.min_box_sizes:
            found = {"pixel": set(), "cv": set()}
            boxes_all = {"pixel": 0, "cv": 0}
            boxes_tp = {"pixel": 0, "cv": 0}
            clean_fp = {"pixel": 0, "cv": 0}
            clean_flagged = {"pixel": 0, "cv": 0}
            for pair in pairs:
                result = compare_images.compare_ui_screenshots(
                    pair["baseline"], pair["actual"], out / "_diffs" / pair["folder"] / f"diff_{pair['page']}.png",
                    threshold=threshold, min_box_size=min_box, ignore_top=args.mask_top, ignore_bottom=args.mask_bottom,
                )
                for method in ("pixel", "cv"):
                    boxes = result[method]["boxes"]
                    boxes_all[method] += len(boxes)
                    boxes_tp[method] += sum(1 for b in boxes if any(_intersects(b, d["bbox"]) for d in pair["defects"]))
                    for defect in pair["defects"]:
                        if any(_intersects(b, defect["bbox"]) for b in boxes):
                            found[method].add(f"{defect['set']}:{defect['id']}")
                    if not pair["defects"]:
                        clean_fp[method] += len(boxes)
                        clean_flagged[method] += bool(boxes)
            for method in ("pixel", "cv"):
                rows.append({
                    "threshold": threshold, "min_box_size": min_box, "method": method,
                    "defects_found": len(found[method]), "defects_total": len(defects_total),
                    "recall": round(len(found[method]) / len(defects_total), 4) if defects_total else None,
                    "boxes": boxes_all[method], "boxes_tp": boxes_tp[method],
                    "box_precision": round(boxes_tp[method] / boxes_all[method], 4) if boxes_all[method] else None,
                    "fp_boxes_clean_pages": clean_fp[method], "clean_pages_flagged": clean_flagged[method],
                })
            print(f"τ={threshold:>3} min_box={min_box:>3} | " + " | ".join(
                f"{r['method']}: {r['defects_found']}/{r['defects_total']} found, {r['boxes']} boxes, {r['fp_boxes_clean_pages']} on clean pages"
                for r in rows[-2:]))

    with open(out / "calibration.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Rank by defects found, box precision, and proximity to defaults (38, 14)
    recommendation = {}
    for method in ("pixel", "cv"):
        candidates = [r for r in rows if r["method"] == method and r["fp_boxes_clean_pages"] <= args.max_clean_fp]
        candidates.sort(key=lambda r: (-r["defects_found"], -(r["box_precision"] or 0),
                                       abs(r["threshold"] - 38) + abs(r["min_box_size"] - 14)))
        recommendation[method] = candidates[0] if candidates else None

    summary = {"sets": args.set, "pages": args.pages, "folders": args.folders, "pairs": len(pairs),
               "pages_missing": skipped, "max_clean_fp": args.max_clean_fp, "recommendation": recommendation,
               "note": "Do not report detection results on the pairs used for this calibration."}
    (out / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\n⚠️ Values calibrated here must be evaluated on other pairs. Results: {out / 'calibration.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
