#!/usr/bin/env python3
"""E2: Local VLM classification of diff bounding boxes (version vs. language comparison)."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import compare_images  # noqa: E402
from common_agent import diff_triage  # noqa: E402

FF3 = ROOT_DIR / "ff3"
FOLDERS = ["baseline_DE_DE", "baseline_EN_US", "baseline_ZH_CN", "baseline_ZH_TW"]
PAGES = ["06_vehicle_tab", "07_settings_tab", "01_home_tab", "language_button", "login_field", "13_language_choice"]


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


def collect_boxes(out):
    gt = json.loads((FF3 / "seeded_defects.json").read_text(encoding="utf-8"))["defects"]
    items = []
    # V: B5 vs. A1 per folder and page
    for folder in FOLDERS:
        for page in PAGES:
            b, a = find_page(FF3 / "runs/A1/screenshots" / folder, page), find_page(FF3 / "runs/B5/screenshots" / folder, page)
            if not b or not a:
                continue
            res = compare_images.compare_ui_screenshots(b, a, out / "_diffs" / "V" / folder / f"diff_{page}.png", threshold=38, min_box_size=14)
            defects = [d for d in gt if d["folder"] == folder and d["page"] == page and d.get("bbox") and any(d["bbox"])]
            for box in res["cv"]["boxes"]:
                hit = [d["id"] for d in defects if intersects(box, d["bbox"])]
                items.append({"set": "V", "folder": folder, "page": page, "baseline": str(b), "actual": str(a),
                              "box": list(box), "defect_ids": hit})
    # L: Cross-language comparison in A1 (DE as reference)
    for folder in FOLDERS[1:]:
        for page in PAGES:
            b, a = find_page(FF3 / "runs/A1/screenshots" / FOLDERS[0], page), find_page(FF3 / "runs/A1/screenshots" / folder, page)
            if not b or not a:
                continue
            res = compare_images.compare_ui_screenshots(b, a, out / "_diffs" / "L" / folder / f"diff_{page}.png", threshold=38, min_box_size=14)
            for box in res["cv"]["boxes"]:
                items.append({"set": "L", "folder": folder, "page": page, "baseline": str(b), "actual": str(a),
                              "box": list(box), "defect_ids": []})
    return items


def rate_all(items, out):
    from PIL import Image
    path = out / "e2_ratings.jsonl"
    done = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[(r["set"], r["folder"], r["page"], tuple(r["box"]))] = r
    model = diff_triage.default_triage_model()
    cache = {}
    with open(path, "a", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            key = (it["set"], it["folder"], it["page"], tuple(it["box"]))
            if key in done:
                it.update(done[key])
                continue
            pair = (it["baseline"], it["actual"])
            if pair not in cache:
                cache.clear()
                base = Image.open(pair[0]).convert("RGB")
                act = Image.open(pair[1]).convert("RGB")
                cache[pair] = (base, act if act.size == base.size else act.resize(base.size))
            base, act = cache[pair]
            verdict = diff_triage.rate_box(base, act, it["box"], model, margin=24)
            it.update(verdict=verdict["verdict"], category=verdict["category"], reason=verdict["reason"], model=model)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(items)} {it['set']} {it['folder']} {it['page']} {it['box']} -> {it['verdict']}", flush=True)
    return items


def share(k, n):
    return f"{k}/{n} ({k / n:.1%})" if n else "0/0"


def summarize(items):
    V = [i for i in items if i["set"] == "V"]
    L = [i for i in items if i["set"] == "L"]
    vd = [i for i in V if i["defect_ids"]]
    vn = [i for i in V if not i["defect_ids"]]
    review = lambda i: i["verdict"] in ("defect", "unclear")
    defects = sorted({d for i in vd for d in i["defect_ids"]})
    kept_defect = sorted({d for i in vd if i["verdict"] == "defect" for d in i["defect_ids"]})
    kept_review = sorted({d for i in vd if review(i) for d in i["defect_ids"]})
    skipped = [i for i in items if i["verdict"] == "acceptable"]
    return {
        "boxes": {"V_total": len(V), "V_defect_boxes": len(vd), "V_other_boxes": len(vn), "L_total": len(L)},
        "verdicts": {s: dict(Counter(i["verdict"] for i in items if i["set"] == s)) for s in ("V", "L")},
        "1_defect_boxes_rated_defect": share(sum(i["verdict"] == "defect" for i in vd), len(vd)),
        "1_defect_boxes_rated_defect_or_unclear": share(sum(review(i) for i in vd), len(vd)),
        "2_defects_with_box_defect": share(len(kept_defect), len(defects)),
        "2_defects_with_box_defect_or_unclear": share(len(kept_review), len(defects)),
        "2_defects_lost_if_only_defect_or_unclear_reviewed": sorted(set(defects) - set(kept_review)),
        "3_V_other_boxes_acceptable": share(sum(i["verdict"] == "acceptable" for i in vn), len(vn)),
        "3_L_boxes_acceptable": share(sum(i["verdict"] == "acceptable" for i in L), len(L)),
        "4_boxes_skippable_acceptable": share(len(skipped), len(items)),
        "4_defects_lost_when_skipping": sorted(set(defects) - set(kept_review)),
        "defect_boxes_detail": [{k: i[k] for k in ("folder", "page", "box", "defect_ids", "verdict", "category", "reason")} for i in vd],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = collect_boxes(out)
    print(f"{len(items)} boxes: V {sum(i['set'] == 'V' for i in items)}, L {sum(i['set'] == 'L' for i in items)}")
    items = rate_all(items, out)
    summary = summarize(items)
    (out / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "defect_boxes_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
