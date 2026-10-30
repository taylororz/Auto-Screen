#!/usr/bin/env python3
"""E5h: Test whether SoM predictions follow element ID numbers or bounding box positions under renumbering."""
import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
STATES = ROOT_DIR / "Qwen_trained_e5" / "diff_results" / "states"
IMG_DIR = ROOT_DIR / "eval_ext" / "results" / "e5h"
INSTRUCTION = "Click the triage location dropdown menu in the upper-middle area"
CASES = {
    "English": "baseline_EN_US/triage_dropdown_20261028_195830_520",
    "German": "baseline_DE_DE/triage_dropdown_20261028_195727_346",
    "中文（简体）": "baseline_ZH_CN/triage_dropdown_20261028_195853_287",
    "中文（繁體）": "baseline_ZH_TW/triage_dropdown_20261028_195957_935",
}
TRAINING_ID = {"English": 9, "German": 8, "中文（简体）": 8, "中文（繁體）": 10}
SEED = 2910
N_PERM = 20


def draw(image, boxes, labels):
    """Draw numbered bounding boxes onto the screenshot."""
    marked = image.copy()
    for (x, y, w, h), number in zip(boxes, labels):
        label = f"[{number}]"
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label_origin = (x, max(18, y - 6))
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(marked, (label_origin[0], label_origin[1] - label_h - 5),
                      (label_origin[0] + label_w + 6, label_origin[1] + 3), (255, 255, 255), -1)
        cv2.putText(marked, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return marked


def elements_for(boxes, labels):
    return [{"id": n, "bbox": {"x": x, "y": y, "width": w, "height": h},
             "center": {"x": int(x + w / 2), "y": int(y + h / 2)}, "area": int(w * h)}
            for (x, y, w, h), n in zip(boxes, labels)]


def roles(boxes):
    """Assign semantic roles to boxes by position (dropdown, label above, lower card header)."""
    role = {}
    for i, (x, y, w, h) in enumerate(boxes):
        if w > 900 and h > 150 and 1000 <= y <= 1150:
            role[i] = "auswahlliste"
        elif 1000 <= y <= 1040 and h < 80:
            role[i] = "beschriftung"
        elif 1630 <= y <= 1680 and h < 120:
            role[i] = "ueberschrift_unten"
    return role


def conditions(boxes):
    n = len(boxes)
    base = list(range(1, n + 1))
    role = roles(boxes)
    inv = {v: k for k, v in role.items()}
    out = [("A0", boxes, base)]
    swap = base[:]
    a, b = inv["auswahlliste"], inv["ueberschrift_unten"]
    swap[a], swap[b] = swap[b], swap[a]
    out.append(("A", boxes, swap))
    rng = random.Random(SEED)
    for k in range(1, N_PERM + 1):
        perm = base[:]
        rng.shuffle(perm)
        out.append((f"B{k:02d}", boxes, perm))
    i_lab, i_dd = inv["beschriftung"], inv["auswahlliste"]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[i_lab], boxes[i_dd]
    ux, uy = min(x1, x2), min(y1, y2)
    union = (ux, uy, max(x1 + w1, x2 + w2) - ux, max(y1 + h1, y2 + h2) - uy)
    merged = [bx for i, bx in enumerate(boxes) if i not in (i_lab, i_dd)] + [union]
    merged = sorted(merged, key=lambda bx: (bx[1], bx[0]))
    out.append(("C", merged, list(range(1, len(merged) + 1))))
    return out


def cmd_run(a):
    spec = importlib.util.spec_from_file_location("visual_agent", ROOT_DIR / "Qwen_trained" / "visual_agent.py")
    va = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(va)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    with open(out, "w", encoding="utf-8") as f:
        for lang, stem in CASES.items():
            raw = cv2.imread(str(STATES / f"{stem}.png"))
            stored = cv2.imread(str(STATES / f"{stem}_som.png"))
            meta = json.loads((STATES / f"{stem}_som.json").read_text(encoding="utf-8"))
            boxes = [(e["bbox"]["x"], e["bbox"]["y"], e["bbox"]["width"], e["bbox"]["height"]) for e in meta["elements"]]
            assert [e["id"] for e in meta["elements"]] == list(range(1, len(boxes) + 1))
            for cond, bxs, labels in conditions(boxes):
                marked = draw(raw, bxs, labels)
                if cond == "A0":
                    identical = stored is not None and marked.shape == stored.shape and not np.any(marked != stored)
                    if not identical:
                        raise SystemExit(f"A0 is not pixel-identical for {lang}")
                img = IMG_DIR / f"{stem.split('/')[0][9:]}_{cond}.png"
                cv2.imwrite(str(img), marked)
                elements = elements_for(bxs, labels)
                res = va.ask_ai_for_element_id(str(img), INSTRUCTION, elements, language=lang)
                tid = res.get("target_id")
                idx = labels.index(tid) if tid in labels else None
                role = roles(bxs).get(idx, "andere") if idx is not None else "ungueltig"
                row = {"language": lang, "condition": cond, "labels": labels, "boxes": bxs, "target_id": tid,
                       "chosen_box_index": idx, "chosen_box": list(bxs[idx]) if idx is not None else None,
                       "chosen_role": role, "training_id": TRAINING_ID[lang], "status": res.get("status"),
                       "raw": str(res.get("rationale"))[:120], "time_ms": round(res.get("time_ms", 0)),
                       "model_path": va.MODEL_PATH, "image": str(img)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(lang, cond, "->", tid, role, flush=True)
    print("done:", out)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    res = {}
    for lang in CASES:
        r = [x for x in rows if x["language"] == lang]
        perm = [x for x in r if x["condition"].startswith("B")]
        box_counts = Counter(x["chosen_role"] for x in perm)
        same_box = Counter(x["chosen_box_index"] for x in perm).most_common(1)[0][1] if perm else 0
        num_train = sum(1 for x in perm if x["target_id"] == x["training_id"])
        if num_train >= 15:
            verdict = "Nummer"
        elif same_box >= 15:
            verdict = "Box"
        else:
            verdict = "gemischt"
        res[lang] = {
            "A0": next(x["chosen_role"] + f" ({x['target_id']})" for x in r if x["condition"] == "A0"),
            "A_tausch": next(x["chosen_role"] + f" ({x['target_id']})" for x in r if x["condition"] == "A"),
            "C_zusammengelegt": next(x["chosen_role"] + f" ({x['target_id']})" for x in r if x["condition"] == "C"),
            "B_gewaehlte_box": dict(box_counts),
            "B_haeufigste_box_von_20": same_box,
            "B_trainingsnummer_von_20": num_train,
            "B_entscheidung": verdict,
            "fallback_or_error": sum(1 for x in r if x["status"] != "analyzed_local"),
        }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    s = sp.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[args.cmd](args)
