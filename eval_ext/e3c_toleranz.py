#!/usr/bin/env python3
"""E3c: Sensitivity analysis across distance tolerances (20 to 100 px) and MUIAnno target boxes."""
import argparse
import glob
import json
import math
from pathlib import Path

ANNOTATIONS = Path("/Users/taylorzhuang/screenshot/iOS-1K-Mobile-UI-Dataset/annotations")
REVIEW_FILES = ["/Users/taylorzhuang/Auto-Screen/*/diff_results/finetune_dataset_DG.jsonl",
                "/Users/taylorzhuang/Masterarbeit/Auto-Screen/*/diff_results/finetune_dataset_DG.jsonl"]
TOLERANCES = (20, 27, 30, 40, 50, 60, 80, 100)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def ff2_check():
    out = {}
    for pattern in REVIEW_FILES:
        for p in sorted(glob.glob(pattern)):
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            moved = sum(1 for r in rows if r.get("gt_x") is not None and r.get("center_x") is not None
                        and math.dist((r["center_x"], r["center_y"]), (r["gt_x"], r["gt_y"])) > 0)
            manual = sum(1 for r in rows if r.get("gt_source") == "manual_click")
            out[p.replace("/Users/taylorzhuang/", "")] = {"records": len(rows), "manual_click": manual,
                                                          "distance_gt_0": moved}
    return out


def ea3_sensitivity(rows):
    cache = {}

    def rect(r):
        if r["image"] not in cache:
            data = json.loads((ANNOTATIONS / f"{r['image']}.json").read_text(encoding="utf-8"))
            cache[r["image"]] = [e["bbox"] for e in data["annotations"]["elements"] if e.get("bbox")]
        gx, gy = r["gt"]
        cands = [b for b in cache[r["image"]]
                 if abs(b["x"] + b["w"] / 2 - gx) <= 3 and abs(b["y"] + b["h"] / 2 - gy) <= 3]
        return cands[0] if len(cands) == 1 else None

    dist = lambda p, g: math.inf if not p else math.dist(p, g)
    inside = lambda p, b: bool(p) and b["x"] <= p[0] <= b["x"] + b["w"] and b["y"] <= p[1] <= b["y"] + b["h"]
    matched = [(r, b) for r, b in ((r, rect(r)) for r in rows) if b]
    res = {"cases": len(rows), "matched": len(matched), "by_tolerance": {}}
    for t in TOLERANCES:
        a = [dist(r.get("point_a"), r["gt"]) <= t for r in rows]
        # cases without refinement keep the first point, as in e3_dg_subimage.py
        b = [dist(r.get("point_b") or r.get("point_a"), r["gt"]) <= t for r in rows]
        only_a = sum(x and not y for x, y in zip(a, b))
        only_b = sum(y and not x for x, y in zip(a, b))
        near = [dist(r.get("point_a"), r["gt"]) <= t for r, _ in matched]
        ins = [inside(r.get("point_a"), bx) for r, bx in matched]
        res["by_tolerance"][str(t)] = {
            "acc_without": round(sum(a) / len(a), 4), "acc_with": round(sum(b) / len(b), 4),
            "only_without_hit": only_a, "only_with_hit": only_b,
            "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
            "agreement_with_rectangle": sum(x == y for x, y in zip(near, ins)),
            "agreement_share": round(sum(x == y for x, y in zip(near, ins)) / len(matched), 4),
            "within_t_outside_rectangle": sum(x and not y for x, y in zip(near, ins)),
            "beyond_t_inside_rectangle": sum(y and not x for x, y in zip(near, ins)),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.run, encoding="utf-8") if l.strip()]
    res = {"ff2_review_files": ff2_check(), "ea3": ea3_sensitivity(rows)}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
