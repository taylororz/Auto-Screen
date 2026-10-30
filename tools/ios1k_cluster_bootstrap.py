#!/usr/bin/env python3
"""Per-image cluster bootstrap for comparing base vs. LoRA SoM accuracy on iOS-1K."""
import collections
import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DRAWS = 10_000
SEED = 20261027


def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    base = {r["case"]: r for r in load(RESULTS / "base.jsonl")}
    lora = {r["case"]: r for r in load(RESULTS / "lora.jsonl")}
    cases = sorted(set(base) & set(lora))
    by_image = collections.defaultdict(list)
    for c in cases:
        by_image[base[c]["image"]].append((base[c]["correct_model"], lora[c]["correct_model"]))
    images = list(by_image)
    n_all = len(cases)
    point = 100 * (sum(l for v in by_image.values() for _, l in v)
                   - sum(b for v in by_image.values() for b, _ in v)) / n_all
    random.seed(SEED)
    diffs = []
    for _ in range(DRAWS):
        sample = [random.choice(images) for _ in images]
        n = sum(len(by_image[i]) for i in sample)
        b = sum(x for i in sample for x, _ in by_image[i])
        l = sum(y for i in sample for _, y in by_image[i])
        diffs.append(100 * (l - b) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * DRAWS)], diffs[int(0.975 * DRAWS) - 1]
    print(f"Instructions: {n_all}, Images: {len(images)}")
    print(f"Difference (LoRA minus Base): {point:.2f} percentage points")
    print(f"95% CI (Cluster bootstrap, {DRAWS} draws, seed {SEED}): {lo:.2f} to {hi:.2f}")
    print(f"Share of draws with difference <= 0: {sum(d <= 0 for d in diffs) / DRAWS:.4f}")


if __name__ == "__main__":
    main()
