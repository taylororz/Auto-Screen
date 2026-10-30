#!/usr/bin/env python3
"""E6: Compare Direct Grounding predictions at 1280 px vs. original resolution on 598 flow steps."""
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "Qwen_trained" / "diff_results" / "finetune_dataset_DG.jsonl"
TOLERANCE = 40
DG_MAX_PIXELS = 1024 * 28 * 28
SIZES = ("1280", "original")
KATAPP_STEPS = {"home_tab", "triage_dropdown", "triage_option", "scan_tab", "overview_tab", "vehicle_tab", "settings_tab",
                "activation_token_button", "login_field", "login_save_button", "language_button", "language_choice",
                "initial_login_save_button"}


def app_of(row):
    return "KatApp" if row["step_name"] in KATAPP_STEPS else "Bluesky"


def set_size(size):
    os.environ["LOCAL_MAX_IMAGE_EDGE"] = "1280" if size == "1280" else "0"


def cmd_run(a):
    for key in ("DG_SUBIMAGE_REFINE", "DG_SUBIMAGE_OVERLAP", "DG_SUBIMAGE_MAX_SHIFT", "LOCAL_DG_MAX_PIXELS"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT_DIR / "Qwen_trained"))
    from transformers import AutoProcessor
    import visual_agent_DG as dg

    dg.get_local_model_and_processor()
    processors = {
        "1280": dg._processor,
        "original": AutoProcessor.from_pretrained(dg.MODEL_PATH, min_pixels=256 * 28 * 28),
    }
    rows = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(json.loads(l)["case"], json.loads(l)["size"]) for l in open(out, encoding="utf-8") if l.strip()}
        print(f"Resuming: {len(done)} calls already completed")
    print(f"{len(rows)} steps, model {dg.MODEL_PATH}")

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            case = f"{r['timestamp']}|{r['step_name']}"
            cx, cy = int(r.get("gt_x", r["center_x"])), int(r.get("gt_y", r["center_y"]))
            for size in SIZES:
                if (case, size) in done:
                    continue
                set_size(size)
                dg._processor = processors[size]
                (x, y), dec = dg.predict_target_coordinates(r["screenshot_path"], r["instruction"])
                row = {"case": case, "index": i, "size": size, "app": app_of(r), "language": r["language"],
                       "step_name": r["step_name"], "instruction": r["instruction"], "screenshot_path": r["screenshot_path"],
                       "confirmed_point": [cx, cy], "point_2809": r["decision"].get("point"), "point": [x, y],
                       "status": dec.get("status"), "raw": str(dec.get("raw_output", dec.get("reason", "")))[:300],
                       "time_ms": round(dec.get("time_ms", 0))}
                row["shift"] = round(math.hypot(x - cx, y - cy), 1)
                row["same_as_2809"] = row["point"] == row["point_2809"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(rows)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    os.environ.pop("LOCAL_MAX_IMAGE_EDGE", None)
    print("done:", out)


def cmd_grid(a):
    """Inspect image grid dimensions and token counts across resolution settings."""
    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0, str(ROOT_DIR / "Qwen_trained"))
    import visual_agent_DG as dg

    rows = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()]
    img = Image.open(rows[0]["screenshot_path"]).convert("RGB")
    conditions = [
        ("SoM, max. 1600 px", 1600, AutoProcessor.from_pretrained(dg.MODEL_PATH)),
        ("SoM, Original", 0, AutoProcessor.from_pretrained(dg.MODEL_PATH)),
        ("DG, max. 1280 px", 1280, AutoProcessor.from_pretrained(dg.MODEL_PATH, min_pixels=256 * 28 * 28,
                                                                  max_pixels=DG_MAX_PIXELS)),
        ("DG, Original", 0, AutoProcessor.from_pretrained(dg.MODEL_PATH, min_pixels=256 * 28 * 28)),
    ]
    res = {"input": list(img.size)}
    for name, edge, proc in conditions:
        im = img
        if edge and max(im.size) > edge:
            s = edge / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.Resampling.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "x"}]}]
        image_inputs, _ = process_vision_info(msgs)
        enc = proc(text=[proc.apply_chat_template(msgs, tokenize=False)], images=image_inputs, return_tensors="pt")
        t, h, w = [int(v) for v in enc["image_grid_thw"][0]]
        patch = proc.image_processor.patch_size
        merge = proc.image_processor.merge_size
        res[name] = {"passed": list(im.size), "processed_px": [w * patch, h * patch], "image_tokens": t * h * w // (merge * merge)}
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_review(a):
    """Export review images for points with > 40 px distance to the verified target point."""
    import cv2
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if r["shift"] > TOLERANCE]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E6 (DG): Manual Review", "",
             "Red = verified point, blue = predicted point in E6.",
             "Set true or false for each case in manual.json.", ""]
    manual_path = out / "manual.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    for r in todo:
        key = f"{r['index']}_{r['size']}"
        img = cv2.imread(r["screenshot_path"])
        (cx, cy), (fx, fy) = r["confirmed_point"], r["point"]
        for (x, y), color in (((cx, cy), (0, 0, 230)), ((fx, fy), (230, 90, 0))):
            cv2.circle(img, (int(x), int(y)), 34, color, 6)
            cv2.line(img, (int(x) - 50, int(y)), (int(x) + 50, int(y)), color, 4)
            cv2.line(img, (int(x), int(y) - 50), (int(x), int(y) + 50), color, 4)
        cv2.putText(img, r["instruction"][:70], (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        name = f"case_{key}.png"
        cv2.imwrite(str(out / name), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        lines.append(f"- `{name}` ({r['app']}, {r['language']}, {r['step_name']}, {r['size']}, distance {r['shift']} px): "
                     f"{r['instruction']}")
        manual.setdefault(key, None)
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manual_path.write_text(json.dumps(manual, indent=2) + "\n", encoding="utf-8")
    print(f"{len(todo)} cases exported for manual review in {out}")


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    manual = json.loads(Path(a.manual).read_text(encoding="utf-8")) if a.manual and Path(a.manual).exists() else {}
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case"]][r["size"]] = r
    paired = {c: v for c, v in by_case.items() if all(s in v for s in SIZES)}

    def hit(r):
        if r["shift"] <= TOLERANCE:
            return True
        return manual.get(f"{r['index']}_{r['size']}")

    res = {"steps_paired": len(paired), "tolerance_px": TOLERANCE}
    for size in SIZES:
        for app in ("KatApp", "Bluesky", "gesamt"):
            R = [v[size] for v in paired.values() if app == "gesamt" or v[size]["app"] == app]
            if not R:
                continue
            shifts = sorted(r["shift"] for r in R)
            hits = [hit(r) for r in R]
            res[f"{size}|{app}"] = {
                "n": len(R),
                "within_tolerance": sum(r["shift"] <= TOLERANCE for r in R),
                "manual_true": sum(1 for r, h in zip(R, hits) if r["shift"] > TOLERANCE and h is True),
                "manual_false": sum(1 for r, h in zip(R, hits) if r["shift"] > TOLERANCE and h is False),
                "manual_open": sum(1 for r, h in zip(R, hits) if r["shift"] > TOLERANCE and h is None),
                "same_point_as_2809": sum(r["same_as_2809"] for r in R),
                "median_shift_px": statistics.median(shifts),
                "fallback_or_error": sum(r["status"] != "analyzed_direct_grounding" for r in R),
                "median_time_s": round(statistics.median(r["time_ms"] for r in R) / 1000, 1),
            }
    # Paired comparison: 1280 px vs. original resolution
    b = c = 0
    for v in paired.values():
        h1, h2 = hit(v["1280"]), hit(v["original"])
        if h1 is None or h2 is None:
            continue
        b += bool(h1) and not h2
        c += bool(h2) and not h1
    res["paired"] = {"only_1280_hit": b, "only_original_hit": c}
    if b + c:
        from math import comb
        k, n = min(b, c), b + c
        res["paired"]["mcnemar_exact_p"] = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--out", required=True)
    s = sub.add_parser("grid")
    s.add_argument("--out", required=True)
    s = sub.add_parser("review")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--manual")
    s.add_argument("--out", required=True)
    a = p.parse_args()
    {"run": cmd_run, "grid": cmd_grid, "review": cmd_review, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
