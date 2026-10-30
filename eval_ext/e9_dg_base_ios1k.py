#!/usr/bin/env python3
"""E9: Direct Grounding evaluation of the base model on the 500-case iOS-1K split."""
import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
                 "ebb281ec70b05090aa6165b016eac8ec08e71b17")
E3_RUN = ROOT / "eval_ext/results/e3.jsonl"
ANNOTATIONS = Path("/Users/taylorzhuang/screenshot/iOS-1K-Mobile-UI-Dataset/annotations")
TOLERANCES = (20, 27, 30, 40, 50, 60, 80, 100)


def load_e3():
    spec = importlib.util.spec_from_file_location("e3_dg_subimage", ROOT / "eval_ext/e3_dg_subimage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_run(a):
    for key in ("DG_SUBIMAGE_REFINE", "DG_SUBIMAGE_OVERLAP", "DG_SUBIMAGE_MAX_SHIFT", "LOCAL_MAX_IMAGE_EDGE",
                "LOCAL_DG_MAX_PIXELS", "LOCAL_MODEL_PATH"):
        os.environ.pop(key, None)
    if a.model == "base":
        os.environ["LOCAL_MODEL_PATH"] = BASE_MODEL
    sys.path.insert(0, str(ROOT / "Qwen_trained"))
    import visual_agent_DG as dg
    e3 = load_e3()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in open(out, encoding="utf-8") if l.strip()}
        print(f"Resuming: {len(done)} cases already completed")
    tmp = out.parent / "e9_tmp"
    tmp.mkdir(exist_ok=True)
    cases = e3.clean_cases(500)
    if a.limit:
        cases = cases[:a.limit]
    print("Model:", dg.MODEL_PATH, flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(cases, 1):
            if r["timestamp"] in done:
                continue
            src = Path(r["screenshot_path"]).expanduser()
            img_path = tmp / f"{e3.stem_id(src)}.png"
            shutil.copyfile(src, img_path)
            gx, gy = float(r["gt_x"]), float(r["gt_y"])
            (x, y), decision = dg.predict_target_coordinates(img_path, r["instruction"])
            row = {"case": r["timestamp"], "image": e3.stem_id(src), "instruction": r["instruction"],
                   "element_type": r["step_name"].rsplit("_", 1)[-1], "gt": [gx, gy], "model": a.model,
                   "status": decision.get("status"), "point": [x, y], "time_ms": round(decision.get("time_ms", 0)),
                   "dist": round(math.hypot(x - gx, y - gy), 1)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            for p in tmp.glob(f"{e3.stem_id(src)}*"):
                p.unlink()
            if i % 10 == 0:
                print(f"{i}/{len(cases)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done:", out)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 4), round((c + m) / d, 4)]


def rect_of(case_row, cache):
    if case_row["image"] not in cache:
        data = json.loads((ANNOTATIONS / f"{case_row['image']}.json").read_text(encoding="utf-8"))
        cache[case_row["image"]] = [e["bbox"] for e in data["annotations"]["elements"] if e.get("bbox")]
    gx, gy = case_row["gt"]
    c = [b for b in cache[case_row["image"]] if abs(b["x"] + b["w"] / 2 - gx) <= 3 and abs(b["y"] + b["h"] / 2 - gy) <= 3]
    return c[0] if len(c) == 1 else None


def cmd_summary(a):
    base = {r["case"]: r for r in (json.loads(l) for l in open(a.base, encoding="utf-8") if l.strip())}
    ft = {r["case"]: r for r in (json.loads(l) for l in open(E3_RUN, encoding="utf-8") if l.strip())}
    cases = [c for c in ft if c in base]
    dist = lambda p, g: math.inf if not p else math.dist(p, g)
    res = {"paired_cases": len(cases), "images": len({ft[c]["image"] for c in cases})}

    def hits(t):
        hb = [dist(base[c]["point"], base[c]["gt"]) <= t for c in cases]
        hf = [dist(ft[c]["point_a"], ft[c]["gt"]) <= t for c in cases]
        return hb, hf

    hb, hf = hits(40)
    only_ft = sum(f and not b for b, f in zip(hb, hf))
    only_base = sum(b and not f for b, f in zip(hb, hf))
    n = len(cases)
    res["tolerance_40"] = {"base": sum(hb), "ft": sum(hf), "base_share": round(sum(hb) / n, 4),
                           "ft_share": round(sum(hf) / n, 4), "base_wilson": wilson(sum(hb), n),
                           "ft_wilson": wilson(sum(hf), n), "only_ft": only_ft, "only_base": only_base,
                           "mcnemar_exact_p": round(mcnemar_exact(only_ft, only_base), 4)}
    res["by_tolerance"] = {}
    for t in TOLERANCES:
        b_, f_ = hits(t)
        ob = sum(f and not b for b, f in zip(b_, f_)); bo = sum(b and not f for b, f in zip(b_, f_))
        res["by_tolerance"][str(t)] = {"base": round(sum(b_) / n, 4), "ft": round(sum(f_) / n, 4),
                                       "only_ft": ob, "only_base": bo, "p": round(mcnemar_exact(ob, bo), 4)}
    cache = {}
    matched = [(c, rect_of(ft[c], cache)) for c in cases]
    matched = [(c, b) for c, b in matched if b]
    inside = lambda p, b: bool(p) and b["x"] <= p[0] <= b["x"] + b["w"] and b["y"] <= p[1] <= b["y"] + b["h"]
    ib = [inside(base[c]["point"], b) for c, b in matched]
    if_ = [inside(ft[c]["point_a"], b) for c, b in matched]
    ob = sum(f and not b for b, f in zip(ib, if_)); bo = sum(b and not f for b, f in zip(ib, if_))
    res["inside_rectangle"] = {"matched": len(matched), "base": sum(ib), "ft": sum(if_), "only_ft": ob, "only_base": bo,
                               "mcnemar_exact_p": round(mcnemar_exact(ob, bo), 4)}
    by = defaultdict(lambda: [0, 0, 0])
    for c, b, f in zip(cases, hb, hf):
        by[ft[c]["element_type"]][0] += b
        by[ft[c]["element_type"]][1] += f
        by[ft[c]["element_type"]][2] += 1
    res["by_element_type_40px"] = {k: f"Basis {b}/{m}, LoRA {f}/{m}" for k, (b, f, m) in sorted(by.items())}
    res["base_median_dist"] = statistics.median(base[c]["dist"] for c in cases)
    res["ft_median_dist"] = statistics.median(ft[c]["dist_a"] for c in cases)
    res["base_without_point_or_fallback"] = sum(1 for c in cases if base[c]["status"] != "analyzed_direct_grounding")
    res["base_median_time_s"] = round(statistics.median(base[c]["time_ms"] for c in cases) / 1000, 1)
    if a.ft_check and Path(a.ft_check).exists():
        chk = [json.loads(l) for l in open(a.ft_check, encoding="utf-8") if l.strip()]
        same = sum(1 for r in chk if r["case"] in ft and r["point"] == ft[r["case"]]["point_a"])
        close = sum(1 for r in chk if r["case"] in ft and dist(r["point"], ft[r["case"]]["point_a"]) <= 3)
        res["ft_reproduction"] = {"n": len(chk), "identical_point": same, "within_3px": close}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=["base", "ft"], required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--base", required=True)
    s.add_argument("--ft-check", default=None)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    main()
