#!/usr/bin/env python3
"""E11: Build and score an independent second-rater evaluation sample (HTML sheet + Cohen's kappa)."""
import argparse
import base64
import csv
import io
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DG_FILE = ROOT / "Qwen_trained/diff_results/finetune_dataset_DG.jsonl"
E5E_FILE = ROOT / "eval_ext/results/e5e.jsonl"
SEED = 2026
KATAPP_STEPS = {"home_tab", "triage_dropdown", "triage_option", "scan_tab", "overview_tab", "vehicle_tab", "settings_tab",
                "activation_token_button", "login_field", "login_save_button", "language_button", "language_choice",
                "initial_login_save_button"}
# Steps where differing box IDs still hit the same UI control
SAME_CONTROL_STEPS = {"go_to_Lists_page", "go_to_Saved_page", "home_tab"}
WIDTH = 520


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def encode(img):
    img = img.convert("RGB")
    h = round(img.height * WIDTH / img.width)
    buf = io.BytesIO()
    img.resize((WIDTH, h), Image.Resampling.LANCZOS).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def mark_point(path, x, y):
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    r = max(18, img.width // 40)
    d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=max(4, r // 4))
    d.line((x - 2 * r, y, x + 2 * r, y), fill=(255, 0, 0), width=3)
    d.line((x, y - 2 * r, x, y + 2 * r), fill=(255, 0, 0), width=3)
    return img


def mark_box(path, b):
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
    d.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=max(5, img.width // 200))
    return img


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(SEED)
    items = []

    dg = [r for r in load_jsonl(DG_FILE) if not r.get("discarded")]
    for app in ("KatApp", "Bluesky"):
        pool = [r for r in dg if (r["step_name"] in KATAPP_STEPS) == (app == "KatApp")]
        for r in rnd.sample(pool, 20):
            items.append({"part": "A", "app": app, "case": r["timestamp"], "step_name": r["step_name"],
                          "language": r["language"], "instruction": r["instruction"], "author": "ja",
                          "image": encode(mark_point(r["screenshot_path"], r["center_x"], r["center_y"]))})

    e5e = load_jsonl(E5E_FILE)
    dev = [r for r in e5e if r["target_replay"] != r["ground_truth_id"]]
    ok = [r for r in e5e if r["target_replay"] == r["ground_truth_id"]]
    for r, author in [(r, "ja" if r["step_name"] in SAME_CONTROL_STEPS else "nein") for r in dev] + \
                     [(r, "ja") for r in rnd.sample(ok, 9)]:
        elements = {e["id"]: e for e in json.load(open(r["elements_file"], encoding="utf-8"))}
        raw = r["annotated_image"].replace("_som.png", ".png")
        items.append({"part": "B", "app": "KatApp" if r["step_name"] in KATAPP_STEPS else "Bluesky",
                      "case": r["case"], "step_name": r["step_name"], "language": r["language"],
                      "instruction": r["instruction"], "author": author,
                      "deviation": r["target_replay"] != r["ground_truth_id"],
                      "image": encode(mark_box(raw, elements[r["target_replay"]]["bbox"]))})

    rnd.shuffle(items)
    for i, it in enumerate(items, 1):
        it["nr"] = i
    key = [{k: v for k, v in it.items() if k != "image"} for it in items]
    (out / "schluessel.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cards = "\n".join(
        f"""<section class="card" id="c{it['nr']}">
  <div class="meta">Case {it['nr']} of {len(items)}</div>
  <p class="instr"><b>Instruction:</b> {it['instruction']}</p>
  <img src="data:image/jpeg;base64,{it['image']}" alt="Case {it['nr']}">
  <div class="opts">
    <label><input type="radio" name="f{it['nr']}" value="ja"> yes</label>
    <label><input type="radio" name="f{it['nr']}" value="nein"> no</label>
    <label><input type="radio" name="f{it['nr']}" value="unklar"> unclear</label>
  </div>
</section>""" for it in items)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Second Rater Review</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f4f4f4; color: #111; margin: 0; padding: 16px; }}
main {{ max-width: 600px; margin: 0 auto; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.card img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
.meta {{ color: #666; font-size: 13px; }}
.opts label {{ display: inline-block; margin: 8px 16px 0 0; font-size: 16px; }}
button {{ font-size: 16px; padding: 10px 16px; margin: 8px 0; }}
#status {{ font-size: 14px; color: #444; }}
</style></head><body><main>
<h1>Second Rater Review</h1>
<p>Each case shows an app screenshot, an instruction, and a red mark (point or frame).
<b>Would a tap on the red mark trigger what the instruction asks?</b>
Choose "unclear" if the image does not show it. At the end, press "Download answers".</p>
{cards}
<p id="status"></p>
<button onclick="dl()">Download answers</button>
</main>
<script>
const N = {len(items)};
function answers() {{
  const rows = [["nr", "antwort"]];
  for (let i = 1; i <= N; i++) {{
    const c = document.querySelector('input[name="f' + i + '"]:checked');
    rows.push([i, c ? c.value : ""]);
  }}
  return rows;
}}
function save() {{
  try {{ localStorage.setItem("e11", JSON.stringify(answers())); }} catch (e) {{}}
  const done = answers().slice(1).filter(r => r[1]).length;
  document.getElementById("status").textContent = done + " of " + N + " answered";
}}
function restore() {{
  try {{
    const rows = JSON.parse(localStorage.getItem("e11") || "[]");
    for (const [nr, v] of rows.slice(1)) {{
      if (v) {{ const el = document.querySelector('input[name="f' + nr + '"][value="' + v + '"]'); if (el) el.checked = true; }}
    }}
  }} catch (e) {{}}
  save();
}}
function dl() {{
  const csv = answers().map(r => r.join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = "bewertung.csv";
  a.click();
}}
document.addEventListener("change", save);
restore();
</script></body></html>
"""
    (out / "bewertung.html").write_text(html, encoding="utf-8")
    print(f"{len(items)} cases (A {sum(i['part'] == 'A' for i in items)}, B {sum(i['part'] == 'B' for i in items)}), "
          f"page {(out / 'bewertung.html').stat().st_size / 1e6:.1f} MB")


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 4), round((c + m) / d, 4)]


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    cats = ("ja", "nein")
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def score(key_path, answers_path, out):
    key = {k["nr"]: k for k in json.load(open(key_path, encoding="utf-8"))}
    ans = {int(r["nr"]): r["antwort"].strip() for r in csv.DictReader(open(answers_path, encoding="utf-8"))}
    res = {"answered": sum(1 for v in ans.values() if v), "cases": len(key)}
    a = [ans.get(n, "") for n, k in key.items() if k["part"] == "A"]
    ja = sum(v == "ja" for v in a)
    judged = sum(v in ("ja", "nein") for v in a)
    res["part_A"] = {"n": len(a), "ja": ja, "nein": sum(v == "nein" for v in a), "unklar": sum(v == "unklar" for v in a),
                     "leer": sum(v == "" for v in a), "share_ja_of_judged": round(ja / judged, 4) if judged else None,
                     "wilson_95": wilson(ja, judged)}
    for name, sel in (("part_B_all", lambda k: k["part"] == "B"),
                      ("part_B_deviations", lambda k: k["part"] == "B" and k.get("deviation"))):
        rows = [(k["author"], ans.get(n, "")) for n, k in key.items() if sel(k)]
        pairs = [(x, y) for x, y in rows if y in ("ja", "nein")]
        res[name] = {"n": len(rows), "judged": len(pairs), "unklar": sum(y == "unklar" for _, y in rows),
                     "leer": sum(y == "" for _, y in rows),
                     "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
                     "cohens_kappa": kappa(pairs),
                     "author_ja_rater_nein": sum(x == "ja" and y == "nein" for x, y in pairs),
                     "author_nein_rater_ja": sum(x == "nein" and y == "ja" for x, y in pairs)}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--key", required=True); s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(Path(a.out))
    else:
        score(a.key, a.answers, a.out)


if __name__ == "__main__":
    main()
