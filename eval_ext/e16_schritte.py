#!/usr/bin/env python3
"""E16: Semi-automatic step, language, and loop-consistency check for replication runs."""
import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = datetime.fromisoformat("2026-10-30T01:54:00+00:00")
LOGS = {"FT SoM": ROOT / "Qwen_trained_e16/diff_results/interaction_log.jsonl",
        "FT DG": ROOT / "Qwen_trained_e16/diff_results/interaction_log_DG.jsonl",
        "Basis SoM": ROOT / "Qwen_untrain_e16/diff_results/interaction_log.jsonl"}


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval_ext" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runs(cmp, models=None):
    """Yield steps per model and run_id starting from START."""
    for model, path in LOGS.items():
        if models and model not in models:
            continue
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if datetime.fromisoformat(r["timestamp"]) >= START]
        by_run = defaultdict(list)
        for r in rows:
            r["annotated_image"] = r.get("annotated_image_path") or r.get("screenshot_path")
            r["case"] = f"{r['timestamp']}|{r['step_name']}"
            by_run[r.get("run_id", "?")].append(r)
        for run_id, seg in sorted(by_run.items()):
            seg.sort(key=lambda r: r["timestamp"])
            yield model, cmp.app_of(seg[0]), run_id, seg


def answer_of(r, model):
    """Return model prediction for cross-loop consistency comparison."""
    if model.endswith("SoM"):
        return r.get("target_id")
    x, y = r.get("center_x"), r.get("center_y")
    return None if x is None else (round(x), round(y))


def cmd_check(a):
    cmp = load("e5e_compare")
    e14 = load("e14_cloud_schritte")
    ref = e14.reference(cmp)
    ref_dg = e14.dg_reference(cmp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, manual, lang_cards = [], [], []
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    for model, app, run_id, seg in runs(cmp, models):
        auto = auto_dg = no_ref = outside = 0
        flows = defaultdict(list)
        for r, key in e14.with_occurrence(cmp, seg):
            flows[(r.get("loop_index"), r["language"])].append(r)
            x, y = r.get("center_x"), r.get("center_y")
            boxes = ref.get(key, [])
            if x is not None and boxes and any(e14.inside(x, y, b) for b in boxes):
                auto += 1
                continue
            if x is not None and model.endswith("DG") and any(
                    ((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= 40 for px, py in ref_dg.get(key, [])):
                auto_dg += 1
                continue
            no_ref += not boxes
            outside += bool(boxes)
            manual.append({"model": model, "app": app, "run_id": run_id, "step_name": r["step_name"],
                           "language": r["language"], "occurrence": key[3], "loop_index": r.get("loop_index"),
                           "instruction": r["instruction"], "timestamp": r["timestamp"], "image": r["screenshot_path"],
                           "x": x, "y": y, "reason": "no reference" if not boxes else
                           ("no point" if x is None else "outside reference box")})
        # Count steps with identical answer across all loops per (language, step, occurrence)
        per_step = defaultdict(list)
        for r, key in e14.with_occurrence(cmp, seg):
            per_step[(r["language"], r["step_name"], key[3])].append(answer_of(r, model))
        same = sum(len(set(map(str, v))) == 1 for v in per_step.values())
        # Sample one mid-flow screenshot per flow for language verification
        for (loop, lang), items in sorted(flows.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            mid = items[len(items) // 2]
            lang_cards.append({"model": model, "app": app, "run_id": run_id, "loop_index": loop, "language": lang,
                               "steps": len(items), "image": mid["screenshot_path"]})
        summary.append({"model": model, "app": app, "run_id": run_id, "steps": len(seg),
                        "auto_correct": auto + auto_dg, "auto_box": auto, "auto_dg_40px": auto_dg,
                        "manual_outside_reference": outside, "manual_no_reference": no_ref,
                        "self_healed": sum(bool((r.get("decision") or {}).get("self_healed")) for r in seg),
                        "step_keys": len(per_step), "same_answer_all_loops": same})
    groups = []
    for m in manual:
        key = (m["model"], m["app"], m["run_id"], m["language"], m["step_name"], m["occurrence"])
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
    for i, c in enumerate(lang_cards, 1):
        c["nr"] = i
    (out / "auto.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manuell.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "sprache.json").write_text(json.dumps(lang_cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cards = []
    for i, g in enumerate(groups, 1):
        m = g["members"][0]
        if Path(m["image"]).exists():
            pic = e14.encode(m["image"], m["x"], m["y"]) if m["x"] is not None else e14.encode(m["image"], -100, -100)
        else:
            pic = None
        cards.append(f"""<section class="card"><div class="meta">Step Group {i} of {len(groups)} · {m['model']} · {m['app']} · {m['language']} · {m['step_name']} · applies to {len(g['members'])} step(s)</div>
<p><b>Instruction:</b> {m['instruction']}</p>{'<p><i>No tap point: model selected no valid box.</i></p>' if m['x'] is None else ''}{'<img src="data:image/jpeg;base64,' + pic + '">' if pic else '<p>(no image)</p>'}
<div class="opts"><label><input type="radio" name="s{i}" value="richtig"> correct</label>
<label><input type="radio" name="s{i}" value="falsch"> wrong</label>
<label><input type="radio" name="s{i}" value="abgelehnt"> validly rejected</label>
<label><input type="radio" name="s{i}" value="falsche_seite"> wrong page</label>
<label><input type="radio" name="s{i}" value="unklar"> unclear</label></div></section>""")
    for c in lang_cards:
        pic = e14.encode(c["image"], -100, -100) if Path(c["image"]).exists() else None
        cards.append(f"""<section class="card small"><div class="meta">Language {c['nr']} of {len(lang_cards)} · {c['model']} · {c['app']} · Loop {c['loop_index']} · expected: <b>{c['language']}</b></div>
{'<img src="data:image/jpeg;base64,' + pic + '">' if pic else '<p>(no image)</p>'}
<div class="opts"><label><input type="radio" name="l{c['nr']}" value="ja"> language matches</label>
<label><input type="radio" name="l{c['nr']}" value="nein"> different language</label></div></section>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>E16 Review</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;color:#111;margin:0;padding:16px}}
main{{max-width:600px;margin:0 auto}}.card{{background:#fff;border-radius:8px;padding:16px;margin:16px 0}}
.card img{{width:100%;height:auto;border:1px solid #ccc}}.small img{{width:50%}}.meta{{color:#666;font-size:13px}}
.opts label{{display:inline-block;margin:8px 16px 0 0;font-size:16px}}button{{font-size:16px;padding:10px 16px}}</style></head>
<body><main><h1>E16: Replication Runs</h1>
<p>Steps: Does the red point hit the target of the instruction? Language: Does the image show the expected language? Press "Download answers" when done.</p>
{''.join(cards)}<p id="status"></p><button onclick="dl()">Download answers</button></main>
<script>
const S={len(groups)},L={len(lang_cards)};const KEY="e16_"+{json.dumps(out.name)};
function answers(){{const r=[["art","nr","urteil"]];for(let i=1;i<=S;i++){{const c=document.querySelector('input[name="s'+i+'"]:checked');r.push(["schritt",i,c?c.value:""]);}}
for(let i=1;i<=L;i++){{const c=document.querySelector('input[name="l'+i+'"]:checked');r.push(["sprache",i,c?c.value:""]);}}return r;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(answers()));}}catch(e){{}}document.getElementById("status").textContent=answers().slice(1).filter(r=>r[2]).length+" of "+(S+L);}}
function restore(){{try{{for(const [art,nr,v] of JSON.parse(localStorage.getItem(KEY)||"[]").slice(1)){{if(v){{const el=document.querySelector('input[name="'+(art==="schritt"?"s":"l")+nr+'"][value="'+v+'"]');if(el)el.checked=true;}}}}}}catch(e){{}}save();}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([answers().map(r=>r.join(",")).join("\\n")],{{type:"text/csv"}}));a.download="e16_antworten.csv";a.click();}}
document.addEventListener("change",save);restore();
</script></body></html>"""
    (out / "e16_pruefung.html").write_text(html, encoding="utf-8")
    for s in summary:
        print(s)
    print("manual pending:", len(manual), "steps in", len(groups), "groups, language images:", len(lang_cards))


def cmd_score(a):
    d = Path(a.dir)
    summary = json.load(open(d / "auto.json", encoding="utf-8"))
    manual = json.load(open(d / "manuell.json", encoding="utf-8"))
    langs = json.load(open(d / "sprache.json", encoding="utf-8"))
    ans = {(r["art"], int(r["nr"])): r["urteil"].strip() for r in csv.DictReader(open(a.answers, encoding="utf-8"))}
    for s in summary:
        run = (s["model"], s["app"], s["run_id"])
        mine = [m["group"] for m in manual if (m["model"], m["app"], m["run_id"]) == run]
        verdict = [ans.get(("schritt", g)) for g in mine]
        s["manual_correct"] = verdict.count("richtig")
        s["manual_wrong"] = verdict.count("falsch")
        s["manual_rejected_not_counted"] = verdict.count("abgelehnt")
        s["manual_wrong_page_not_counted"] = verdict.count("falsche_seite")
        s["manual_unclear"] = sum(v in ("unklar", "", None) for v in verdict)
        s["correct_total"] = s["auto_correct"] + s["manual_correct"]
        s["counted_steps"] = s["steps"] - s["manual_rejected_not_counted"] - s["manual_wrong_page_not_counted"]
        lv = [ans.get(("sprache", c["nr"])) for c in langs if (c["model"], c["app"], c["run_id"]) == run]
        s["flows_seen"] = len(lv)
        s["language_ok"] = lv.count("ja")
        s["language_wrong"] = lv.count("nein")
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for s in summary:
        print(f"{s['model']:<10} {s['app']:<8} {s['correct_total']}/{s['counted_steps']} steps "
              f"(auto {s['auto_correct']}, wrong {s['manual_wrong']}, unclear {s['manual_unclear']}), "
              f"language {s['language_ok']}/{s['flows_seen']}, same answer {s['same_answer_all_loops']}/{s['step_keys']}")


def cmd_sprache(a):
    """Export language verification images at a fixed step per flow."""
    cmp = load("e5e_compare")
    e14 = load("e14_cloud_schritte")
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for model, app, run_id, seg in runs(cmp, models):
        if app != a.app:
            continue
        flows = defaultdict(list)
        for r in seg:
            flows[(r.get("loop_index"), r["language"])].append(r)
        for (loop, lang), items in sorted(flows.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            hit = [r for r in items if r["step_name"] == a.step]
            cards.append({"model": model, "app": app, "run_id": run_id, "loop_index": loop, "language": lang,
                          "step": a.step if hit else None, "image": hit[0]["screenshot_path"] if hit else None})
    for i, c in enumerate(cards, 1):
        c["nr"] = i
    (out / "sprache.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_cards = []
    for c in cards:
        pic = e14.encode(c["image"], -100, -100) if c["image"] and Path(c["image"]).exists() else None
        html_cards.append(f"""<section class="card small"><div class="meta">Language {c['nr']} of {len(cards)} · {c['model']} · {c['app']} · Loop {c['loop_index']} · Step {c['step']} · expected: <b>{c['language']}</b></div>
{'<img src="data:image/jpeg;base64,' + pic + '">' if pic else '<p>(step not reached, no image)</p>'}
<div class="opts"><label><input type="radio" name="l{c['nr']}" value="ja"> language matches</label>
<label><input type="radio" name="l{c['nr']}" value="nein"> different language</label>
<label><input type="radio" name="l{c['nr']}" value="unklar"> unreadable</label></div></section>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>E16 Language {a.app}</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;color:#111;margin:0;padding:16px}}
main{{max-width:600px;margin:0 auto}}.card{{background:#fff;border-radius:8px;padding:16px;margin:16px 0}}
.card img{{width:60%;height:auto;border:1px solid #ccc}}.meta{{color:#666;font-size:13px}}
.opts label{{display:inline-block;margin:8px 16px 0 0;font-size:16px}}button{{font-size:16px;padding:10px 16px}}</style></head>
<body><main><h1>E16: Language per Flow ({a.app}, Step {a.step})</h1>
<p>Does the settings screen show the expected language? Press "Download answers" when done.</p>
{''.join(html_cards)}<p id="status"></p><button onclick="dl()">Download answers</button></main>
<script>
const L={len(cards)};
function answers(){{const r=[["art","nr","urteil"]];for(let i=1;i<=L;i++){{const c=document.querySelector('input[name="l'+i+'"]:checked');r.push(["sprache",i,c?c.value:""]);}}return r;}}
function save(){{try{{localStorage.setItem("e16s_{out.name}",JSON.stringify(answers()));}}catch(e){{}}document.getElementById("status").textContent=answers().slice(1).filter(r=>r[2]).length+" of "+L;}}
function restore(){{try{{for(const [art,nr,v] of JSON.parse(localStorage.getItem("e16s_{out.name}")||"[]").slice(1)){{if(v){{const el=document.querySelector('input[name="l'+nr+'"][value="'+v+'"]');if(el)el.checked=true;}}}}}}catch(e){{}}save();}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([answers().map(r=>r.join(",")).join("\\n")],{{type:"text/csv"}}));a.download="e16_sprache_{a.app}.csv";a.click();}}
document.addEventListener("change",save);restore();
</script></body></html>"""
    (out / f"e16_sprache_{a.app}.html").write_text(html, encoding="utf-8")
    print("language images:", len(cards), "without image:", sum(c["image"] is None for c in cards))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--out", required=True)
    c.add_argument("--models", default=None, help='e.g. "FT SoM,FT DG" or "Basis SoM"')
    s = sub.add_parser("score"); s.add_argument("--dir", required=True); s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    l = sub.add_parser("sprache"); l.add_argument("--out", required=True); l.add_argument("--app", required=True)
    l.add_argument("--step", required=True); l.add_argument("--models", default=None)
    a = ap.parse_args()
    {"check": cmd_check, "score": cmd_score, "sprache": cmd_sprache}[a.cmd](a)


if __name__ == "__main__":
    main()
