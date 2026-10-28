#!/usr/bin/env python3
"""Resampling je Bild für den iOS-1K-Vergleich (Basismodell gegen LoRA, SoM).

Liest results/base.jsonl und results/lora.jsonl (je Anweisung ein Eintrag) und
schätzt die Differenz der Treffergenauigkeit des Modells (correct_model) mit einem
Cluster-Bootstrap. Gezogen werden ganze Bilder mit Zurücklegen, weil Anweisungen auf
demselben Bild nicht unabhängig sind. Es werden nur vorhandene Werte gelesen.
Nichts in results/ wird verändert.

Aufruf (aus dem Repository-Wurzelverzeichnis):
    python3 tools/ios1k_cluster_bootstrap.py
"""
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
    print(f"Anweisungen: {n_all}, Bilder: {len(images)}")
    print(f"Differenz (LoRA minus Basis): {point:.2f} Prozentpunkte")
    print(f"95-%-Intervall (Cluster-Bootstrap, {DRAWS} Ziehungen, Seed {SEED}): {lo:.2f} bis {hi:.2f}")
    print(f"Anteil der Ziehungen mit Differenz <= 0: {sum(d <= 0 for d in diffs) / DRAWS:.4f}")


if __name__ == "__main__":
    main()
