#!/usr/bin/env python3
"""E5e: Paired comparison of base vs. fine-tuned model on saved SoM screenshots."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from math import comb
from pathlib import Path

FLOW_GAP_S = 90


def app_of(row):
    folder = row["annotated_image"].split("/states/")[1].split("/")[0]
    return "Bluesky" if folder in ("baseline_DE", "baseline_EN") or row["language"] == "Chinese" else "KatApp"


def flows_of(rows):
    """Split steps by app, language, and time gaps > FLOW_GAP_S into separate flows."""
    groups = defaultdict(list)
    for r in rows:
        groups[(app_of(r), r["language"])].append(r)
    flows = []
    for key, items in groups.items():
        items.sort(key=lambda r: r["case"])
        current, last = [], None
        for r in items:
            t = datetime.fromisoformat(r["case"].split("|")[0])
            if last is not None and (t - last).total_seconds() > FLOW_GAP_S:
                flows.append((key, current))
                current = []
            current.append(r)
            last = t
        flows.append((key, current))
    return flows


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(a):
    base = {r["case"]: r for r in (json.loads(l) for l in open(a.base, encoding="utf-8") if l.strip())}
    lora = {r["case"]: r for r in (json.loads(l) for l in open(a.lora, encoding="utf-8") if l.strip())}
    cases = sorted(set(base) & set(lora))
    rows = [base[c] for c in cases if base[c]["ground_truth_id"] is not None]
    ok_base = {r["case"]: r["target_replay"] == r["ground_truth_id"] for r in rows}
    ok_lora = {c: lora[c]["target_replay"] == lora[c]["ground_truth_id"] for c in ok_base}
    b = sum(ok_lora[c] and not ok_base[c] for c in ok_base)
    c = sum(ok_base[c] and not ok_lora[c] for c in ok_base)

    errors = [r for r in rows if not ok_base[r["case"]]]
    by_app = defaultdict(lambda: [0, 0])
    for r in rows:
        by_app[app_of(r)][0] += ok_base[r["case"]]
        by_app[app_of(r)][1] += 1

    flow_res = defaultdict(lambda: [0, 0])
    first_errors = Counter()
    for (app, lang), items in flows_of(rows):
        wrong = [r for r in items if not ok_base[r["case"]]]
        flow_res[app][0] += not wrong
        flow_res[app][1] += 1
        if wrong:
            first_errors[f"{app}|{wrong[0]['step_name']}"] += 1

    res = {
        "steps_paired": len(ok_base),
        "base_correct": f"{sum(ok_base.values())}/{len(ok_base)}",
        "lora_correct": f"{sum(ok_lora.values())}/{len(ok_lora)}",
        "mcnemar": {"lora_only_correct": b, "base_only_correct": c, "p_exact": mcnemar_exact(b, c)},
        "base_correct_by_app": {k: f"{v[0]}/{v[1]}" for k, v in by_app.items()},
        "base_errors_by_step": dict(Counter(f"{app_of(r)}|{r['step_name']}" for r in errors).most_common()),
        "base_errors_by_language": dict(Counter(f"{app_of(r)}|{r['language']}" for r in errors).most_common()),
        "base_fallback_or_error": sum(1 for r in rows if r["status"] != "analyzed_local"),
        "flows_without_error_base": {k: f"{v[0]}/{v[1]}" for k, v in flow_res.items()},
        "first_error_step_base": dict(first_errors.most_common()),
        "base_error_examples": [{k: r[k] for k in ("case", "language", "step_name", "instruction", "ground_truth_id",
                                                    "target_replay")} for r in errors[:40]],
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "base_error_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
