#!/usr/bin/env python3
"""E13: Build and score error-category review sheet for icon mispredictions on the iOS-1K split."""
import argparse
import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/lora.jsonl"
SOM = ROOT / "datasets/dataset_ios1k/som_images"
CATEGORIES = [("a", "ambiguous instruction"), ("b", "similar icon selected"), ("c", "label occluded or unreadable"),
              ("d", "no valid ID (heuristic fallback)"), ("e", "other"),
              ("f", "same UI control (functionally correct)"),
              ("g", "wrong ground-truth reference, red box also wrong"),
              ("h", "wrong ground-truth reference, red box matches instruction (model correct)")]
WIDTH = 520


def encode(img):
    h = round(img.height * WIDTH / img.width)
    buf = io.BytesIO()
    img.convert("RGB").resize((WIDTH, h), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()]
    cases = [r for r in rows if r["element_type"] == "icon" and not r["correct_model"]]
    cards, meta = [], []
    for i, r in enumerate(cases, 1):
        els = {e["id"]: e for e in json.loads((SOM / f"{r['image']}_som.json").read_text(encoding="utf-8"))["elements"]}
        img = Image.open(SOM / f"{r['image']}_som.png").convert("RGB")
        d = ImageDraw.Draw(img)
        rad = max(26, img.width // 22)
        for eid, col in ((r["gt_id"], (0, 170, 0)), (r.get("pred_id"), (255, 0, 0))):
            if eid in els:
                c = els[eid]["center"]
                d.ellipse((c["x"] - rad, c["y"] - rad, c["x"] + rad, c["y"] + rad), outline=col, width=6)
        if r["fallback"]:
            note = (f"Model produced no valid ID (raw: {r.get('raw', '')[:60]}). Red box was selected by heuristic "
                    f"fallback, not the model. Ground truth is #{r['gt_id']} (green). -> (d)")
        else:
            note = f"Model selected #{r['pred_id']} (red), ground truth is #{r['gt_id']} (green)."
        meta.append({"nr": i, "case": r["case"], "image": r["image"], "instruction": r["instruction"],
                     "gt_id": r["gt_id"], "pred_id": r["pred_id"], "fallback": r["fallback"]})
        opts = "".join(f'<label><input type="radio" name="f{i}" value="{k}"> ({k}) {t}</label>' for k, t in CATEGORIES)
        cards.append(f"""<section class="card"><div class="meta">Case {i} of {len(cases)} · Image {r['image']}</div>
<p><b>Instruction:</b> {r['instruction']}</p><p class="note">{note}</p>
<img src="data:image/jpeg;base64,{encode(img)}" alt="Case {i}"><div class="opts">{opts}</div></section>""")
    (out / "faelle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Icon Misses</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;color:#111;margin:0;padding:16px}}
main{{max-width:600px;margin:0 auto}}.card{{background:#fff;border-radius:8px;padding:16px;margin:16px 0}}
.card img{{width:100%;height:auto;border:1px solid #ccc}}.meta{{color:#666;font-size:13px}}.note{{color:#333;font-size:14px}}
.opts label{{display:block;margin:6px 0;font-size:15px}}button{{font-size:16px;padding:10px 16px;margin:8px 0}}</style></head>
<body><main><h1>E13: Icon Misses</h1>
<p>Select one cause per case. Red = model selection, green = reference box. Press "Download answers" when done.</p>
{''.join(cards)}
<p id="status"></p><button onclick="dl()">Download answers</button></main>
<script>
const N={len(cases)};
function answers(){{const r=[["nr","ursache"]];for(let i=1;i<=N;i++){{const c=document.querySelector('input[name="f'+i+'"]:checked');r.push([i,c?c.value:""]);}}return r;}}
function save(){{try{{localStorage.setItem("e13",JSON.stringify(answers()));}}catch(e){{}}
document.getElementById("status").textContent=answers().slice(1).filter(r=>r[1]).length+" of "+N+" assigned";}}
function restore(){{try{{for(const [nr,v] of JSON.parse(localStorage.getItem("e13")||"[]").slice(1)){{if(v){{const el=document.querySelector('input[name="f'+nr+'"][value="'+v+'"]');if(el)el.checked=true;}}}}}}catch(e){{}}save();}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([answers().map(r=>r.join(",")).join("\\n")],{{type:"text/csv"}}));a.download="e13_antworten.csv";a.click();}}
document.addEventListener("change",save);restore();
</script></body></html>"""
    (out / "symbole.html").write_text(html, encoding="utf-8")
    print(f"{len(cases)} cases, fallback {sum(r['fallback'] for r in cases)}, "
          f"page {(out / 'symbole.html').stat().st_size / 1e6:.1f} MB")


def score(cases_path, answers_path, out):
    cases = {c["nr"]: c for c in json.load(open(cases_path, encoding="utf-8"))}
    ans = {int(r["nr"]): r["ursache"].strip() for r in csv.DictReader(open(answers_path, encoding="utf-8"))}
    counts = Counter(ans.get(n, "") for n in cases)
    res = {"cases": len(cases), "by_category": {f"({k}) {t}": counts.get(k, 0) for k, t in CATEGORIES},
           "unassigned": counts.get("", 0),
           "fallback_cases_not_d": sum(1 for n, c in cases.items() if c["fallback"] and ans.get(n) != "d")}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--cases", required=True); s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    build(Path(a.out)) if a.cmd == "build" else score(a.cases, a.answers, a.out)


if __name__ == "__main__":
    main()
