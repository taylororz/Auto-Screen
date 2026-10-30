#!/usr/bin/env python3
"""E14: Semi-automatic step verification for cloud model evaluation runs."""
import argparse
import base64
import csv
import importlib.util
import io
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OLD = Path("/Users/taylorzhuang/Masterarbeit/Auto-Screen")
LOGS = {"Sol SoM": ROOT / "openAI_6.0_Sol/diff_results/interaction_log.jsonl",
        "Luna SoM": OLD / "openAI_6.0_Luna/diff_results/interaction_log.jsonl",
        "Claude SoM": ROOT / "claude/diff_results/interaction_log.jsonl",
        "Claude DG": ROOT / "claude/diff_results/interaction_log_DG.jsonl"}
# (model, app, local start time) -> (condition, expected steps)
RUNS = {("Sol SoM", "KatApp", "29.09. 23:51"): ("Original", 262), ("Sol SoM", "Bluesky", "30.09. 00:48"): ("Original", 336),
        ("Sol SoM", "KatApp", "06.10. 07:54"): ("1600", 262), ("Sol SoM", "Bluesky", "05.10. 23:31"): ("1600", 336),
        ("Luna SoM", "KatApp", "30.09. 02:37"): ("Original", 262), ("Luna SoM", "Bluesky", "30.09. 03:01"): ("Original", 336),
        ("Luna SoM", "KatApp", "05.10. 15:03"): ("1600", 262), ("Luna SoM", "Bluesky", "05.10. 19:55"): ("1600", 317),
        ("Claude SoM", "KatApp", "03.10. 21:16"): ("Original", 262), ("Claude SoM", "Bluesky", "05.10. 10:07"): ("Original", 336),
        ("Claude SoM", "KatApp", "05.10. 13:20"): ("1600", 262), ("Claude SoM", "Bluesky", "05.10. 13:44"): ("1600", 318),
        ("Claude DG", "KatApp", "03.10. 20:08"): ("Original", 262), ("Claude DG", "Bluesky", "03.10. 20:30"): ("Original", 336),
        ("Claude DG", "KatApp", "05.10. 12:02"): ("1600", 262), ("Claude DG", "Bluesky", "05.10. 12:33"): ("1600", 336)}
WIDTH = 520


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval_ext" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def local(ts):
    return datetime.fromisoformat(ts) + timedelta(hours=2)


def with_occurrence(cmp, rows):
    """Attach step occurrence index within each flow (grouped by app, language, and 90s gap)."""
    out = []
    for (app, lang), items in cmp.flows_of(rows):
        seen = defaultdict(int)
        for r in items:
            seen[r["step_name"]] += 1
            out.append((r, (app, lang, r["step_name"], seen[r["step_name"]])))
    return out


def reference(cmp):
    e5d = load("e5d_som_replay")
    steps = [s for s in e5d.load_steps() if s["ground_truth_id"] is not None]
    ref = defaultdict(list)
    for s, key in with_occurrence(cmp, steps):
        els = json.loads(Path(s["elements_file"]).read_text(encoding="utf-8"))
        els = els if isinstance(els, list) else els["elements"]
        ref[key].append(next(e["bbox"] for e in els if e["id"] == s["ground_truth_id"]))
    return ref


def dg_reference(cmp):
    """Load verified Direct Grounding points from the fine-tuned dataset."""
    rows = [json.loads(l) for l in open(ROOT / "Qwen_trained/diff_results/finetune_dataset_DG.jsonl", encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if not r.get("discarded")]
    for r in rows:
        r["annotated_image"] = r["screenshot_path"]
        r["case"] = f"{r['timestamp']}|{r['step_name']}"
    ref = defaultdict(list)
    for r, key in with_occurrence(cmp, rows):
        ref[key].append((r["gt_x"], r["gt_y"]))
    return ref


def runs(cmp):
    for model, path in LOGS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            r["annotated_image"] = r.get("annotated_image_path") or r.get("screenshot_path")
            r["case"] = f"{r['timestamp']}|{r['step_name']}"
        by_app = defaultdict(list)
        for r in rows:
            by_app[cmp.app_of(r)].append(r)
        for app, rs in by_app.items():
            rs.sort(key=lambda r: r["timestamp"])
            segs, seg, last = [], [], None
            for r in rs:
                t = datetime.fromisoformat(r["timestamp"])
                if last and t - last > timedelta(minutes=15):
                    segs.append(seg)
                    seg = []
                seg.append(r)
                last = t
            segs.append(seg)
            for seg in segs:
                key = (model, app, f"{local(seg[0]['timestamp']):%d.%m. %H:%M}")
                if key not in RUNS:
                    continue
                cond, expected = RUNS[key]
                if len(seg) > expected:
                    ts = [datetime.fromisoformat(r["timestamp"]) for r in seg]
                    cut = max(range(len(ts) - 1), key=lambda i: ts[i + 1] - ts[i]) + 1
                    seg = seg[cut:]
                assert len(seg) == expected, f"{key}: {len(seg)} vs {expected}"
                yield model, app, cond, key[2], seg


def inside(x, y, b):
    return b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]


def encode(path, x, y):
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    r = max(18, img.width // 40)
    d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=max(4, r // 4))
    d.line((x - 2 * r, y, x + 2 * r, y), fill=(255, 0, 0), width=3)
    d.line((x, y - 2 * r, x, y + 2 * r), fill=(255, 0, 0), width=3)
    h = round(img.height * WIDTH / img.width)
    buf = io.BytesIO()
    img.resize((WIDTH, h), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def cmd_check(a):
    cmp = load("e5e_compare")
    ref = reference(cmp)
    ref_dg = dg_reference(cmp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, manual = [], []
    for model, app, cond, start, seg in runs(cmp):
        auto = auto_dg = no_ref = outside = 0
        for r, key in with_occurrence(cmp, seg):
            x, y = r.get("center_x"), r.get("center_y")
            boxes = ref.get(key, [])
            if x is not None and boxes and any(inside(x, y, b) for b in boxes):
                auto += 1
                continue
            if x is not None and model.endswith("DG") and any(
                    ((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= 40 for px, py in ref_dg.get(key, [])):
                auto_dg += 1
                continue
            no_ref += not boxes
            outside += bool(boxes)
            manual.append({"model": model, "app": app, "condition": cond, "run_start": start, "step_name": r["step_name"],
                           "language": r["language"], "occurrence": key[3], "instruction": r["instruction"],
                           "timestamp": r["timestamp"], "image": r["screenshot_path"], "x": x, "y": y,
                           "reason": "no reference" if not boxes else
                           ("no point" if x is None else "outside reference box")})
        summary.append({"model": model, "app": app, "condition": cond, "run_start": start, "steps": len(seg),
                        "auto_correct": auto + auto_dg, "auto_box": auto, "auto_dg_40px": auto_dg,
                        "manual_outside_reference": outside, "manual_no_reference": no_ref})
    # Group steps sharing the same key with points within 10 px
    groups = []
    for m in manual:
        key = (m["model"], m["condition"], m["app"], m["run_start"], m["language"], m["step_name"], m["occurrence"])
        for g in groups:
            if g["key"] == key and m["x"] is not None and g["x"] is not None and \
                    ((m["x"] - g["x"]) ** 2 + (m["y"] - g["y"]) ** 2) ** 0.5 <= 10:
                g["members"].append(m)
                break
        else:
            groups.append({"key": key, "x": m["x"], "y": m["y"], "members": [m]})
    for i, g in enumerate(groups, 1):
        for m in g["members"]:
            m["group"] = i
    (out / "auto.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manuell.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cards = []
    for i, g in enumerate(groups, 1):
        m = g["members"][0]
        img = m["image"] if Path(m["image"]).exists() else m["image"].replace(str(ROOT) + "/", str(OLD) + "/")
        pic = encode(img, m["x"], m["y"]) if m["x"] is not None and Path(img).exists() else None
        m = dict(m, nr=i)
        cards.append(f"""<section class="card"><div class="meta">Group {i} of {len(groups)} · {m['app']} · {m['language']} · applies to {len(g['members'])} step(s)</div>
<p><b>Instruction:</b> {m['instruction']}</p>{'<img src="data:image/jpeg;base64,' + pic + '">' if pic else '<p>(no image or point)</p>'}
<div class="opts"><label><input type="radio" name="f{m['nr']}" value="richtig"> correct</label>
<label><input type="radio" name="f{m['nr']}" value="falsch"> wrong</label>
<label><input type="radio" name="f{m['nr']}" value="unklar"> unclear</label></div></section>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cloud Steps</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;color:#111;margin:0;padding:16px}}
main{{max-width:600px;margin:0 auto}}.card{{background:#fff;border-radius:8px;padding:16px;margin:16px 0}}
.card img{{width:100%;height:auto;border:1px solid #ccc}}.meta{{color:#666;font-size:13px}}
.opts label{{display:inline-block;margin:8px 16px 0 0;font-size:16px}}button{{font-size:16px;padding:10px 16px}}</style></head>
<body><main><h1>E14: Cloud Steps Pending Manual Review</h1>
<p>Does the red tap point hit the target of the instruction? Press "Download answers" when done.</p>
{''.join(cards)}<p id="status"></p><button onclick="dl()">Download answers</button></main>
<script>
const N={len(groups)};
function answers(){{const r=[["nr","urteil"]];for(let i=1;i<=N;i++){{const c=document.querySelector('input[name="f'+i+'"]:checked');r.push([i,c?c.value:""]);}}return r;}}
function save(){{try{{localStorage.setItem("e14",JSON.stringify(answers()));}}catch(e){{}}document.getElementById("status").textContent=answers().slice(1).filter(r=>r[1]).length+" of "+N;}}
function restore(){{try{{for(const [nr,v] of JSON.parse(localStorage.getItem("e14")||"[]").slice(1)){{if(v){{const el=document.querySelector('input[name="f'+nr+'"][value="'+v+'"]');if(el)el.checked=true;}}}}}}catch(e){{}}save();}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([answers().map(r=>r.join(",")).join("\\n")],{{type:"text/csv"}}));a.download="e14_antworten.csv";a.click();}}
document.addEventListener("change",save);restore();
</script></body></html>"""
    (out / "cloud_schritte.html").write_text(html, encoding="utf-8")
    for s in summary:
        print(s)
    print("manual pending:", len(manual), "steps in", len(groups), "groups")


def cmd_score(a):
    d = Path(a.dir)
    summary = json.load(open(d / "auto.json", encoding="utf-8"))
    manual = json.load(open(d / "manuell.json", encoding="utf-8"))
    ans = {int(r["nr"]): r["urteil"].strip() for r in csv.DictReader(open(a.answers, encoding="utf-8"))}
    for s in summary:
        mine = [m["group"] for m in manual if (m["model"], m["app"], m["condition"], m["run_start"]) ==
                (s["model"], s["app"], s["condition"], s["run_start"])]
        s["manual_correct"] = sum(ans.get(g) == "richtig" for g in mine)
        s["manual_wrong"] = sum(ans.get(g) == "falsch" for g in mine)
        # Correctly rejected steps (target missing or unboxed), excluded from denominator
        s["manual_rejected_not_counted"] = sum(ans.get(g) == "abgelehnt" for g in mine)
        s["manual_unclear"] = sum(ans.get(g) in ("unklar", "", None) for g in mine)
        s["correct_total"] = s["auto_correct"] + s["manual_correct"]
        s["counted_steps"] = s["steps"] - s["manual_rejected_not_counted"]
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for s in summary:
        print(f"{s['model']:<11} {s['condition']:<8} {s['app']:<8} {s['correct_total']}/{s['counted_steps']} "
              f"(auto {s['auto_correct']}, wrong {s['manual_wrong']}, rejected {s['manual_rejected_not_counted']}, "
              f"unclear {s['manual_unclear']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--dir", required=True); s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_check(a) if a.cmd == "check" else cmd_score(a)


if __name__ == "__main__":
    main()
