#!/usr/bin/env python3
"""E15: Build and score review sheet for Simplenote transfer runs (DG and SoM)."""
import argparse
import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SINCE = "2026-10-29T22:36"
LOGS = {"DG": ROOT / "Qwen_trained/diff_results/interaction_log_DG.jsonl",
        "SoM": ROOT / "Qwen_trained/diff_results/interaction_log.jsonl"}
WIDTH = 480


def box_of(row, tid):
    path = Path(row["annotated_image_path"]).with_suffix(".json")
    if tid is None or not path.exists():
        return None
    els = json.loads(path.read_text(encoding="utf-8"))
    els = els if isinstance(els, list) else els.get("elements", [])
    return next((e for e in els if str(e["id"]) == str(tid)), None)


def picture(row, mode):
    img = Image.open(row["screenshot_path"]).convert("RGB")
    d = ImageDraw.Draw(img)
    w = max(5, img.width // 200)
    if mode == "DG":
        x, y = row["center_x"], row["center_y"]
        r = max(18, img.width // 40)
        d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=w)
        d.line((x - 2 * r, y, x + 2 * r, y), fill=(255, 0, 0), width=3)
        d.line((x, y - 2 * r, x, y + 2 * r), fill=(255, 0, 0), width=3)
    else:
        model_id = (row.get("decision") or {}).get("target_id")
        for tid, col in ((row.get("target_id"), (0, 90, 255)), (model_id, (255, 0, 0))):
            e = box_of(row, tid)
            if e and not (col == (0, 90, 255) and str(tid) == str(model_id)):
                b = e["bbox"]
                d.rectangle((b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]), outline=col, width=w)
    h = round(img.height * WIDTH / img.width)
    buf = io.BytesIO()
    img.resize((WIDTH, h), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    groups = []
    for mode, path in LOGS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in (x for x in rows if x["timestamp"] >= SINCE):
            d = r.get("decision") or {}
            model_id = d.get("target_id")
            healed = mode == "SoM" and str(model_id) != str(r.get("target_id"))
            key = (mode, r["language"], r["step_name"], str(model_id) if mode == "SoM" else "")
            for g in groups:
                if g["key"] == key and abs(g["x"] - r["center_x"]) <= 10 and abs(g["y"] - r["center_y"]) <= 10 \
                        and g["healed"] == healed:
                    g["members"].append(r["timestamp"])
                    break
            else:
                groups.append({"key": key, "mode": mode, "language": r["language"], "step_name": r["step_name"],
                               "instruction": r["instruction"], "x": r["center_x"], "y": r["center_y"],
                               "model_id": model_id, "tapped_id": r.get("target_id"), "healed": healed,
                               "members": [r["timestamp"]], "row": r})
    cards = []
    for i, g in enumerate(groups, 1):
        g["nr"] = i
        note = ("Engine saw no screen change after tapping the red box and then tapped the blue candidate."
                if g["healed"] else "")
        cards.append(f"""<section class="card"><div class="meta">Group {i} of {len(groups)} · {g['mode']} · {g['language']} ·
{g['step_name']} · applies to {len(g['members'])} step(s)</div><p><b>Instruction:</b> {g['instruction']}</p>
<p class="note">{note}</p><img src="data:image/jpeg;base64,{picture(g['row'], g['mode'])}">
<div class="opts"><label><input type="radio" name="f{i}" value="richtig"> red correct</label>
<label><input type="radio" name="f{i}" value="falsch"> red wrong</label>
<label><input type="radio" name="f{i}" value="unklar"> unclear</label>
<label><input type="radio" name="f{i}" value="falsche_seite"> wrong page, not counted</label></div></section>""")
    meta = [{k: v for k, v in g.items() if k not in ("row", "key")} for g in groups]
    (out / "gruppen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Simplenote</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;color:#111;margin:0;padding:16px}}
main{{max-width:560px;margin:0 auto}}.card{{background:#fff;border-radius:8px;padding:16px;margin:16px 0}}
.card img{{width:100%;height:auto;border:1px solid #ccc}}.meta{{color:#666;font-size:13px}}.note{{color:#0050d0;font-size:14px}}
.opts label{{display:inline-block;margin:8px 16px 0 0;font-size:16px}}button{{font-size:16px;padding:10px 16px}}</style></head>
<body><main><h1>E15: Simplenote</h1><p>Red = model choice (DG: point, SoM: box), blue = second candidate tapped by
engine. Rate only red. Press "Download answers" when done.</p>{''.join(cards)}
<p id="status"></p><button onclick="dl()">Download answers</button></main><script>
const N={len(groups)};
function answers(){{const r=[["nr","urteil"]];for(let i=1;i<=N;i++){{const c=document.querySelector('input[name="f'+i+'"]:checked');r.push([i,c?c.value:""]);}}return r;}}
function save(){{try{{localStorage.setItem("e15",JSON.stringify(answers()));}}catch(e){{}}document.getElementById("status").textContent=answers().slice(1).filter(r=>r[1]).length+" of "+N;}}
function restore(){{try{{for(const [nr,v] of JSON.parse(localStorage.getItem("e15")||"[]").slice(1)){{if(v){{const el=document.querySelector('input[name="f'+nr+'"][value="'+v+'"]');if(el)el.checked=true;}}}}}}catch(e){{}}save();}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([answers().map(r=>r.join(",")).join("\\n")],{{type:"text/csv"}}));a.download="e15_antworten.csv";a.click();}}
document.addEventListener("change",save);restore();</script></body></html>"""
    (out / "simplenote.html").write_text(html, encoding="utf-8")
    c = Counter((g["mode"], g["healed"]) for g in groups)
    print(f"{len(groups)} groups, {sum(len(g['members']) for g in groups)} steps", dict(c))


def score(d, answers, out):
    groups = json.load(open(Path(d) / "gruppen.json", encoding="utf-8"))
    ans = {int(r["nr"]): r["urteil"].strip() for r in csv.DictReader(open(answers, encoding="utf-8"))}
    res = {}
    for mode in ("DG", "SoM"):
        G = [g for g in groups if g["mode"] == mode]
        n = sum(len(g["members"]) for g in G)
        ok = sum(len(g["members"]) for g in G if ans.get(g["nr"]) == "richtig")
        # Exclude steps where the app was on the wrong screen (e.g. after failed login)
        off = sum(len(g["members"]) for g in G if ans.get(g["nr"]) == "falsche_seite")
        res[mode] = {"steps": n, "counted_steps": n - off, "wrong_page_not_counted": off, "model_choice_correct": ok,
                     "wrong": sum(len(g["members"]) for g in G if ans.get(g["nr"]) == "falsch"),
                     "unclear_or_open": sum(len(g["members"]) for g in G
                                            if ans.get(g["nr"]) not in ("richtig", "falsch", "falsche_seite")),
                     "healed_steps": sum(len(g["members"]) for g in G if g["healed"])}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--dir", required=True); s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    build(Path(a.out)) if a.cmd == "build" else score(a.dir, a.answers, a.out)


if __name__ == "__main__":
    main()
