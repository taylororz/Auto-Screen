#!/usr/bin/env python3
"""
FF3-Auswertung: Erkennung gezielt eingebrachter Darstellungsfehler im Versionsvergleich.

Nutzt die Vergleichsfunktionen von Auto-Screen (compare_images.py) UNVERÄNDERT.
Die von den beiden Verfahren gefundenen Boxen werden abgegriffen, indem
_draw_diff_highlight() zur Laufzeit umhüllt wird. Dadurch werden exakt dieselben
Boxen ausgewertet, die auch im HTML-Bericht erscheinen.

Beispiele
  # Fehlerbuild B gegen Referenzlauf A1
  python3 ff3_eval.py --tool-dir ~/Auto-Screen \
      --baseline runs/A1/screenshots --actual runs/B1/screenshots \
      --gt seeded_defects.json --out results/B1_vs_A1

  # Kontrolle: zweiter Lauf von A gegen A1 (keine Fehler eingebracht)
  python3 ff3_eval.py --tool-dir ~/Auto-Screen \
      --baseline runs/A1/screenshots --actual runs/A2/screenshots \
      --out results/A2_vs_A1
"""
import argparse, csv, importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path

PIXEL_COLOR = (0, 200, 0)   # compare_pixel_matching
CV_COLOR = (0, 0, 240)      # compare_cv_grayscale
DEFAULT_FOLDERS = ["baseline_DE_DE", "baseline_EN_US", "baseline_ZH_CN", "baseline_ZH_TW"]
DEFAULT_PAGES = ["triage_location_menu", "triage_option_page", "scan_page", "overview_page",
                 "vehicle_page", "settings_page", "activation_token_page", "login_page",
                 "language_dialog", "language_choice_page"]


def load_tool(tool_dir):
    path = Path(tool_dir).expanduser() / "compare_images.py"
    spec = importlib.util.spec_from_file_location("compare_images", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    captured = {}
    original = mod._draw_diff_highlight

    def wrapped(image, boxes, color=(0, 0, 255)):
        captured[tuple(color)] = [tuple(int(v) for v in b) for b in boxes]
        return original(image, boxes, color)

    mod._draw_diff_highlight = wrapped
    return mod, captured


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-dir", required=True)
    ap.add_argument("--baseline", required=True, help="screenshots-Ordner des Referenzlaufs")
    ap.add_argument("--actual", required=True, help="screenshots-Ordner des Vergleichslaufs")
    ap.add_argument("--gt", default=None, help="seeded_defects.json (leer lassen für Kontrolllauf)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS)
    ap.add_argument("--pages", nargs="*", default=DEFAULT_PAGES)
    ap.add_argument("--threshold", type=int, default=38)
    ap.add_argument("--min-box-size", type=int, default=14)
    args = ap.parse_args()

    mod, captured = load_tool(args.tool_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    gt = []
    if args.gt:
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))["defects"]
        for d in gt:
            assert d["folder"] in args.folders, f"{d['id']}: Ordner {d['folder']} nicht ausgewertet"
            assert d["page"] in args.pages, f"{d['id']}: Seite {d['page']} nicht ausgewertet"
    gt_by_page = defaultdict(list)
    for d in gt:
        gt_by_page[(d["folder"], d["page"])].append(d)

    rows, missing = [], []
    detected = {m: set() for m in ("pixel", "cv")}
    for folder in args.folders:
        for page in args.pages:
            b = find_page(Path(args.baseline) / folder, page)
            a = find_page(Path(args.actual) / folder, page)
            if not b or not a:
                missing.append((folder, page, bool(b), bool(a)))
                continue
            captured.clear()
            mod.compare_ui_screenshots(b, a, out / "diffs" / folder / f"diff_{page}.png",
                                       threshold=args.threshold, min_box_size=args.min_box_size)
            boxes = {"pixel": captured.get(PIXEL_COLOR, []), "cv": captured.get(CV_COLOR, [])}
            defects = gt_by_page.get((folder, page), [])
            for method, bx in boxes.items():
                tp_boxes = [x for x in bx if any(intersects(x, d["bbox"]) for d in defects)]
                for d in defects:
                    if any(intersects(x, d["bbox"]) for x in bx):
                        detected[method].add(d["id"])
                rows.append({
                    "folder": folder, "page": page, "method": method,
                    "has_defect": int(bool(defects)), "n_defects": len(defects),
                    "n_boxes": len(bx), "tp_boxes": len(tp_boxes), "fp_boxes": len(bx) - len(tp_boxes),
                    "boxes": json.dumps(bx),
                })

    with open(out / "page_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["folder"])
        w.writeheader(); w.writerows(rows)

    summary = {"baseline": args.baseline, "actual": args.actual, "gt": args.gt,
               "pages_compared": len(rows) // 2, "pages_missing": missing, "methods": {}}
    for method in ("pixel", "cv"):
        R = [r for r in rows if r["method"] == method]
        tp_b = sum(r["tp_boxes"] for r in R); all_b = sum(r["n_boxes"] for r in R)
        pos = [r for r in R if r["has_defect"]]; neg = [r for r in R if not r["has_defect"]]
        m = {
            "defects_total": len(gt),
            "defects_detected": len(detected[method]),
            "defect_recall": len(detected[method]) / len(gt) if gt else None,
            "boxes_total": all_b, "boxes_tp": tp_b, "boxes_fp": all_b - tp_b,
            "box_precision": tp_b / all_b if all_b else None,
            "pages_with_defect": len(pos),
            "pages_with_defect_flagged": sum(1 for r in pos if r["n_boxes"] > 0),
            "pages_without_defect": len(neg),
            "pages_without_defect_flagged": sum(1 for r in neg if r["n_boxes"] > 0),
            "fp_boxes_on_clean_pages": sum(r["n_boxes"] for r in neg),
            "recall_by_class": {},
            "missed_defects": sorted(d["id"] for d in gt if d["id"] not in detected[method]),
        }
        by_cls = defaultdict(lambda: [0, 0])
        for d in gt:
            by_cls[d["class"]][1] += 1
            by_cls[d["class"]][0] += d["id"] in detected[method]
        m["recall_by_class"] = {k: f"{a}/{n}" for k, (a, n) in sorted(by_cls.items())}
        summary["methods"][method] = m

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    if missing:
        print(f"\nWARNUNG: {len(missing)} Seiten fehlen in einem der Läufe:", missing, file=sys.stderr)


if __name__ == "__main__":
    main()
