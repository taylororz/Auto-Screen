#!/usr/bin/env python3
"""E10: Cross-language screenshot comparison (EN vs. DE) on FF3 runs A1, A2, and B5."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FF3 = ROOT / "ff3"
RUNS = ("A1", "A2", "B5")
REF, LANG = "baseline_EN_US", "baseline_DE_DE"
PAGE_ALIASES = {"language_button": "12_language_button", "login_field": "09_login_field"}
NOT_CLEARLY_VISIBLE = {"D02", "D05", "D09", "D13", "D11", "D14"}


def load_ff3_eval():
    spec = importlib.util.spec_from_file_location("ff3_eval", FF3 / "ff3_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def steps_of(folder):
    return sorted(p.stem for p in folder.glob("*.png") if re.match(r"^\d{2}_", p.stem))


def crop_equal(img_a, img_b, bbox):
    x, y, w, h = bbox
    return list(img_a.crop((x, y, x + w, y + h)).getdata()) == list(img_b.crop((x, y, x + w, y + h)).getdata())


def defect_steps(defects, b5_de, a1_de, variant):
    """Return B5 (DE) steps showing a defect ('annotated' step only, or all steps where defect is visibly 'shown')."""
    steps = steps_of(b5_de)
    images = {s: Image.open(b5_de / f"{s}.png").convert("RGB") for s in steps}
    a1 = {s: Image.open(a1_de / f"{s}.png").convert("RGB") for s in steps_of(a1_de)}
    result = {}
    for d in defects:
        page = PAGE_ALIASES.get(d["page"], d["page"])
        assert page in steps, f"{d['id']}: {page} missing"
        hits = {page}
        if variant == "shown":
            for s in steps:
                if (s != page and s in a1 and images[s].size == images[page].size == a1[s].size
                        and crop_equal(images[s], images[page], d["bbox"])
                        and not crop_equal(images[s], a1[s], d["bbox"])):
                    hits.add(s)
        for s in hits:
            result.setdefault(s, set()).add(d["id"])
    return result


def units_of(run_dir):
    """Steps with identical DE image form one unit, represented by the lowest step number."""
    groups = {}
    for s in steps_of(run_dir / LANG):
        groups.setdefault(md5(run_dir / LANG / f"{s}.png"), []).append(s)
    return [sorted(v) for v in groups.values()]


def precision_at_k(values, labels):
    k = sum(labels)
    if k == 0:
        return None
    order = sorted(zip(values, labels), key=lambda t: -t[0])
    v = order[k - 1][0]
    above = [l for x, l in order if x > v]
    tied = [l for x, l in order if x == v]
    slots = k - len(above)
    expected = sum(above) + (slots * sum(tied) / len(tied) if tied else 0)
    return round(expected / k, 4)


def auc(values, labels):
    pos = [x for x, l in zip(values, labels) if l]
    neg = [x for x, l in zip(values, labels) if not l]
    if not pos or not neg:
        return None
    score = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(score / (len(pos) * len(neg)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ff3_eval = load_ff3_eval()
    mod, captured = ff3_eval.load_tool(str(ROOT))
    defects = json.loads((FF3 / "seeded_defects.json").read_text(encoding="utf-8"))["defects"]
    b5_de = FF3 / "runs" / "B5" / "screenshots" / LANG
    a1_de = FF3 / "runs" / "A1" / "screenshots" / LANG
    diff_dir = Path(a.out).parent / "e10_diffs"

    counts = {}
    for run in RUNS:
        run_dir = FF3 / "runs" / run / "screenshots"
        for unit in sorted(units_of(run_dir), key=lambda u: u[0]):
            step = unit[0]
            captured.clear()
            mod.compare_ui_screenshots(run_dir / REF / f"{step}.png", run_dir / LANG / f"{step}.png",
                                       diff_dir / run / f"diff_{step}.png", threshold=38, min_box_size=14)
            counts[(run, tuple(unit))] = (len(captured.get(ff3_eval.PIXEL_COLOR, [])),
                                          len(captured.get(ff3_eval.CV_COLOR, [])))

    result = {"reference": REF, "language": LANG, "variants": {}}
    for variant in ("annotated", "shown"):
        shown = defect_steps(defects, b5_de, a1_de, variant)
        out = {"defect_steps_B5": {k: sorted(v) for k, v in sorted(shown.items())}, "runs": {}}
        per_step = {}
        for run in RUNS:
            rows = []
            for (r, unit), (px, cv) in counts.items():
                if r != run:
                    continue
                ids = set().union(*(shown.get(s, set()) for s in unit))
                rows.append({"unit": list(unit), "pixel": px, "cv": cv, "defects": sorted(ids),
                             "defects_clearly_visible": sorted(ids - NOT_CLEARLY_VISIBLE)})
                per_step[(run, unit[0])] = rows[-1]
            metrics = {}
            for method in ("pixel", "cv"):
                vals = [x[method] for x in rows]
                for name, key in (("all", "defects"), ("clearly_visible", "defects_clearly_visible")):
                    labels = [bool(x[key]) for x in rows]
                    metrics[f"{method}_{name}"] = {"units": len(rows), "defect_units": sum(labels),
                                                   "precision_at_k": precision_at_k(vals, labels),
                                                   "auc": auc(vals, labels)}
            out["runs"][run] = {"units": rows, "metrics": metrics}
        deltas = []
        for x in out["runs"]["B5"]["units"]:
            a1x = per_step.get(("A1", x["unit"][0]))
            if a1x:
                deltas.append({"step": x["unit"][0], "defect_unit": bool(x["defects"]),
                               "clearly_visible": bool(x["defects_clearly_visible"]),
                               "pixel_B5_minus_A1": x["pixel"] - a1x["pixel"], "cv_B5_minus_A1": x["cv"] - a1x["cv"]})
        out["B5_minus_A1"] = deltas
        result["variants"][variant] = out

    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for variant, out in result["variants"].items():
        print("==", variant, out["defect_steps_B5"])
        for run in RUNS:
            print(run, json.dumps(out["runs"][run]["metrics"], ensure_ascii=False))
        print(json.dumps(out["B5_minus_A1"], ensure_ascii=False))

if __name__ == "__main__":
    main()
