#!/usr/bin/env python3
"""
E3, nachträgliche Auswertung (nicht vorab geplant, siehe PROTOKOLL.md unter „Abweichungen“).

Der Evaluationssplit enthält nur die Mittelpunkte der Zielelemente. Die Rechtecke stehen in den
Annotationen von MUIAnno. Jede Anweisung wird über den Mittelpunkt (Abweichung bis 3 px) einem Rechteck
zugeordnet, nur eindeutige Zuordnungen zählen. Treffer = Punkt innerhalb des Rechtecks.
Es werden die vorhandenen Vorhersagen aus e3.jsonl ausgewertet, ohne neue Inferenz.

  python3 eval_ext/e3_boxcheck.py --run eval_ext/results/e3.jsonl --out eval_ext/results/e3_boxcheck.json
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ANNOTATIONS = Path("/Users/taylorzhuang/screenshot/iOS-1K-Mobile-UI-Dataset/annotations")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.run, encoding="utf-8") if l.strip()]
    cache = {}
    matched, unmatched = [], Counter()
    for r in rows:
        if r["image"] not in cache:
            data = json.loads((ANNOTATIONS / f"{r['image']}.json").read_text(encoding="utf-8"))
            cache[r["image"]] = [e["bbox"] for e in data["annotations"]["elements"] if e.get("bbox")]
        gx, gy = r["gt"]
        cands = [b for b in cache[r["image"]] if abs(b["x"] + b["w"] / 2 - gx) <= 3 and abs(b["y"] + b["h"] / 2 - gy) <= 3]
        if len(cands) == 1:
            matched.append((r, cands[0]))
        else:
            unmatched["kein Rechteck" if not cands else "mehrdeutig"] += 1
    inside = lambda p, b: b["x"] <= p[0] <= b["x"] + b["w"] and b["y"] <= p[1] <= b["y"] + b["h"]
    ha = [inside(r["point_a"], b) for r, b in matched]
    hb = [inside(r["point_b"], b) for r, b in matched]
    only_a = sum(x and not y for x, y in zip(ha, hb))
    only_b = sum(y and not x for x, y in zip(ha, hb))
    by = defaultdict(lambda: [0, 0, 0])
    for (r, _), x, y in zip(matched, ha, hb):
        by[r["element_type"]][0] += x
        by[r["element_type"]][1] += y
        by[r["element_type"]][2] += 1
    n = len(matched)
    res = {
        "cases": len(rows), "matched": n, "unmatched": dict(unmatched),
        "hits_inside_without": sum(ha), "hits_inside_with": sum(hb),
        "acc_without": round(sum(ha) / n, 4), "acc_with": round(sum(hb) / n, 4),
        "only_without_hit": only_a, "only_with_hit": only_b, "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
        "by_element_type": {k: f"ohne {x}/{m}, mit {y}/{m}" for k, (x, y, m) in sorted(by.items())},
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
