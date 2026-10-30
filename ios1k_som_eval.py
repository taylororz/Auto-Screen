#!/usr/bin/env python3
"""FF2: Set-of-Mark evaluation on the deduplicated iOS-1K evaluation split."""
import argparse, importlib.util, json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------- Helper functions
def stem_id(path):
    """'.../0783_som.png' or '.../0783.png' -> '0783'"""
    s = Path(path).stem
    return s[:-4] if s.endswith("_som") else s


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test (binomial) for discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def resolve(path, root):
    p = Path(path).expanduser()
    if p.exists():
        return p
    if root:
        q = Path(root).expanduser() / p.name
        if q.exists():
            return q
    return None


def load_tool(tool_dir):
    path = Path(tool_dir).expanduser() / "visual_agent.py"
    spec = importlib.util.spec_from_file_location("visual_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def elements_for(rec, va, som_path, orig_root, cache_dir):
    """Load saved SoM elements JSON or re-annotate the original screenshot."""
    meta = som_path.with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["elements"], som_path, "meta"
    src = resolve(rec["screenshot_path"], orig_root)
    if src is None:
        return None, None, "image_missing"
    out = Path(cache_dir) / f"{stem_id(src)}_som.png"
    if out.exists() and out.with_suffix(".json").exists():
        els = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["elements"]
    else:
        _, _, els = va.annotate_ui_elements(src, output_path=out)
    return els, out, "reannotated"


# ---------------------------------------------------------------- run
def cmd_run(a):
    va = load_tool(a.tool_dir)
    va.MODEL_PATH = str(Path(a.model).expanduser()) if Path(a.model).expanduser().exists() else a.model
    va._model = va._processor = None

    train_ids = {stem_id(x["images"][0]) for x in json.load(open(a.train, encoding="utf-8"))}
    recs = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    clean = [r for r in recs if stem_id(r["screenshot_path"]) not in train_ids
             and not r.get("discarded", False)]
    print(f"Eval cases: {len(recs)} | Images in training: {len(recs) - len(clean)} cases excluded "
          f"| cleaned: {len(clean)} cases across {len({stem_id(r['screenshot_path']) for r in clean})} images")
    if a.n and a.n < len(clean):
        clean = random.Random(42).sample(clean, a.n)
        print(f"Sample: {len(clean)} cases (seed 42)")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        # keep only successful cases; skipped cases are retried
        old = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        keep = [x for x in old if x["status"] == "ok"]
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep), encoding="utf-8")
        done = {x["case"] for x in keep}
        print(f"Resuming: {len(done)} cases already completed")
    cache = out.parent / "som_cache"; cache.mkdir(exist_ok=True)

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(clean, 1):
            case = r["timestamp"]
            if case in done:
                continue
            row = {"case": case, "image": stem_id(r["screenshot_path"]), "instruction": r["instruction"],
                   "element_type": r["step_name"].rsplit("_", 1)[-1], "gt_id": r["target_id"],
                   "model": a.model}
            som = resolve(r["annotated_image_path"], a.image_root)
            els, som_used, src = elements_for(r, va, som, a.orig_root, cache) if som else (None, None, "som_missing")
            if not els:
                row.update(status="skipped", reason=src)
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue
            # Consistency check: GT ID must exist and its center must match the GT point
            byid = {e["id"]: e for e in els}
            gt = byid.get(r["target_id"])
            if gt is None or abs(gt["center"]["x"] - r["gt_x"]) > 3 or abs(gt["center"]["y"] - r["gt_y"]) > 3:
                row.update(status="skipped", reason=f"gt_mismatch_{src}")
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue

            d = va.ask_ai_for_element_id(str(som_used), r["instruction"], els)
            pred = d.get("target_id")
            row.update(status="ok", element_source=src, n_boxes=len(els), pred_id=pred,
                       decision_status=d.get("status"), fallback=d.get("status") == "analyzed_fallback",
                       correct_system=pred == r["target_id"],
                       correct_model=(pred == r["target_id"]) and d.get("status") != "analyzed_fallback",
                       raw=str(d.get("rationale"))[:200], time_ms=round(d.get("time_ms", 0)))
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0:
                el = time.time() - t0
                print(f"{i}/{len(clean)}  {el/60:.1f} min")
    print("done:", out)


# ---------------------------------------------------------------- summary
def metrics(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    n = len(ok)
    km = sum(r["correct_model"] for r in ok)
    ks = sum(r["correct_system"] for r in ok)
    fb = sum(r["fallback"] for r in ok)
    by = defaultdict(lambda: [0, 0])
    for r in ok:
        by[r["element_type"]][1] += 1
        by[r["element_type"]][0] += r["correct_model"]
    t = sorted(r["time_ms"] for r in ok)
    return {
        "cases_evaluated": n,
        "cases_skipped": Counter(r.get("reason") for r in rows if r["status"] != "ok"),
        "images": len({r["image"] for r in ok}),
        "accuracy_model": round(km / n, 4) if n else None,
        "accuracy_model_ci95": wilson(km, n),
        "accuracy_system_incl_fallback": round(ks / n, 4) if n else None,
        "fallback_rate": round(fb / n, 4) if n else None,
        "accuracy_by_element_type": {k: f"{a}/{b} ({a/b:.1%})" for k, (a, b) in sorted(by.items())},
        "median_time_ms": t[len(t) // 2] if t else None,
    }


def cmd_summary(a):
    runs = {}
    for p in a.runs:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        runs[Path(p).stem] = rows
    res = {name: metrics(rows) for name, rows in runs.items()}
    if len(runs) == 2:
        (na, ra), (nb, rb) = runs.items()
        A = {r["case"]: r for r in ra if r["status"] == "ok"}
        B = {r["case"]: r for r in rb if r["status"] == "ok"}
        common = sorted(set(A) & set(B))
        b = sum(1 for c in common if A[c]["correct_model"] and not B[c]["correct_model"])
        c = sum(1 for c in common if not A[c]["correct_model"] and B[c]["correct_model"])
        res["paired_comparison"] = {
            "common_cases": len(common),
            f"only_{na}_correct": b, f"only_{nb}_correct": c,
            "mcnemar_exact_p": mcnemar_exact(b, c),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--tool-dir", required=True)
    r.add_argument("--eval", required=True)
    r.add_argument("--train", required=True, help="qwen_vl_dataset_combined.json")
    r.add_argument("--model", required=True, help="HF ID of the base model or directory of the LoRA model")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=0)
    r.add_argument("--image-root", default=None, help="Directory of SoM images (*_som.png) if paths do not match")
    r.add_argument("--orig-root", default=None, help="Directory of original images if paths do not match")
    s = sp.add_parser("summary")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summary(a)
