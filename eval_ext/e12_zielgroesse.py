#!/usr/bin/env python3
"""E12: Compute target element bounding-box size distributions across KatApp and Bluesky flows."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCALE = 3


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval_ext" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def describe(sizes):
    short = sorted(min(w, h) for w, h in sizes)
    q5 = short[max(0, round(0.05 * (len(short) - 1)))]
    return {"n": len(sizes), "min_width_px": min(w for w, _ in sizes), "min_height_px": min(h for _, h in sizes),
            "short_side_min_px": short[0], "short_side_q05_px": q5, "short_side_median_px": statistics.median(short),
            "short_side_min_pt": round(short[0] / SCALE, 1), "short_side_median_pt": round(statistics.median(short) / SCALE, 1),
            "share_short_side_below_80px": round(sum(s < 80 for s in short) / len(short), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e5d, cmp = load("e5d_som_replay"), load("e5e_compare")
    sizes, smallest = {"KatApp": [], "Bluesky": []}, []
    for s in e5d.load_steps():
        if s["ground_truth_id"] is None:
            continue
        elements = json.loads(Path(s["elements_file"]).read_text(encoding="utf-8"))
        elements = elements if isinstance(elements, list) else elements["elements"]
        box = next(e["bbox"] for e in elements if e["id"] == s["ground_truth_id"])
        app = cmp.app_of(s)
        sizes[app].append((box["width"], box["height"]))
        smallest.append((min(box["width"], box["height"]), app, s["step_name"], s["language"], box["width"], box["height"]))
    res = {app: describe(v) for app, v in sizes.items()}
    res["gesamt"] = describe(sizes["KatApp"] + sizes["Bluesky"])
    res["smallest_targets"] = [{"short_side_px": x[0], "app": x[1], "step_name": x[2], "language": x[3],
                                "width_px": x[4], "height_px": x[5]} for x in sorted(smallest)[:5]]
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
