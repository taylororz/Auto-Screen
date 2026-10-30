#!/usr/bin/env python3
"""E7: Check how many iOS-1K target box centers match MUIAnno annotation centers within 3 px."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = Path("/Users/taylorzhuang/screenshot/iOS-1K-Mobile-UI-Dataset/annotations")
TOLERANCE = 3


def stem_id(path):
    m = re.search(r"(\d{4})(?:_som)?\.png", path)
    return m.group(1) if m else path


def muianno_centers(stem, cache):
    if stem not in cache:
        data = json.loads((ANNOTATIONS / f"{stem}.json").read_text(encoding="utf-8"))
        cache[stem] = [(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
                       for b in (e.get("bbox") for e in data["annotations"]["elements"]) if b]
    return cache[stem]


def check(records, cache):
    hits = sum(any(abs(r["gt_x"] - x) <= TOLERANCE and abs(r["gt_y"] - y) <= TOLERANCE
                   for x, y in muianno_centers(stem_id(r["screenshot_path"]), cache))
               for r in records)
    return {"n": len(records),
            "images": len({stem_id(r["screenshot_path"]) for r in records}),
            "match_muianno": hits,
            "share": round(hits / len(records), 4) if records else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "datasets/dataset_ios1k/ios1k_som_eval.jsonl"))
    ap.add_argument("--train-records", default=str(ROOT / "datasets/dataset_ios1k/ios1k_som_train.jsonl"))
    ap.add_argument("--train", default=str(ROOT / "qwen_vl_dataset_combined.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    train_ids = {stem_id(x["images"][0]) for x in json.load(open(a.train, encoding="utf-8"))}
    load = lambda p: [r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())
                      if not r.get("discarded", False)]
    # same selection as ios1k_som_eval.py: eval images that are not in the training data
    eval_clean = [r for r in load(a.eval) if stem_id(r["screenshot_path"]) not in train_ids]
    train_used = [r for r in load(a.train_records) if stem_id(r["screenshot_path"]) in train_ids]

    cache = {}
    result = {"tolerance_px": TOLERANCE,
              "evaluation_split": check(eval_clean, cache),
              "training_ios1k": check(train_used, cache)}
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
