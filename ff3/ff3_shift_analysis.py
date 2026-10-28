"""Klassifiziert die Pixel- und Grau-Boxen (B5 gegen A1) als verschoben oder verändert.

Aufruf (aus dem Verzeichnis Auto-Screen):
  python3 ff3/ff3_shift_analysis.py ff3/results ff3/seeded_defects.json ff3/ff3_shift_boxes.json \
      ff3/runs/A1/screenshots ff3/runs/B5/screenshots

Eingaben: page_results.csv aus results/B5_vs_A1 (Boxen), Originalscreenshots A1 und B5 (Ordner baseline_DE_DE).
Die Boxen stammen aus der Auswertung. Die Klassifikation liest nur die Originalbilder.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np

R, GT_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
A_DIR, B_DIR = Path(sys.argv[4]) / "baseline_DE_DE", Path(sys.argv[5]) / "baseline_DE_DE"
gt = json.load(open(GT_PATH, encoding="utf-8"))["defects"]
rows = list(csv.DictReader(open(f"{R}/B5_vs_A1/page_results.csv", encoding="utf-8")))


def inter(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


S = 60  # Suchradius in Pixeln
out = []
for r in rows:
    if r["folder"] != "baseline_DE_DE":
        continue
    m, p = r["method"], r["page"]
    boxes = json.loads(r["boxes"])
    g = [d["bbox"] for d in gt if d["folder"] == r["folder"] and d["page"] == p and d["bbox"] and d["bbox"][2] > 0]
    Ag = cv2.imread(str(A_DIR / f"{p}.png"), cv2.IMREAD_GRAYSCALE)
    Bg = cv2.imread(str(B_DIR / f"{p}.png"), cv2.IMREAD_GRAYSCALE)
    if Ag is None or Bg is None or Ag.shape != Bg.shape:
        sys.exit(f"Screenshot fehlt oder Größe ungleich: {p}")
    H, W = Ag.shape
    for b in boxes:
        x, y, w, h = b
        tp = any(inter(b, q) for q in g)
        res = {"page": p, "method": m, "box": b, "tp": tp}
        t = Bg[y + 3:y + h - 2, x + 3:x + w - 2]
        if t.size == 0 or t.shape[0] < 6 or t.shape[1] < 6 or t.std() < 3:
            res.update(cls="unklar", reason="zu klein/einfarbig")
            out.append(res)
            continue
        x0, y0 = max(0, x + 3 - S), max(0, y + 3 - S)
        x1, y1 = min(W, x + w - 2 + S), min(H, y + h - 2 + S)
        reg = Ag[y0:y1, x0:x1]
        mt = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(mt)
        dx, dy = loc[0] + x0 - (x + 3), loc[1] + y0 - (y + 3)
        z = mt[(y + 3) - y0, (x + 3) - x0]
        a = Ag[loc[1] + y0:loc[1] + y0 + t.shape[0], loc[0] + x0:loc[0] + x0 + t.shape[1]].astype(int)
        mad = float(np.abs(a - t.astype(int)).mean())
        cls = "verschoben" if (dx, dy) != (0, 0) and mx >= 0.95 and mad < 6 else "verändert"
        res.update(cls=cls, dx=int(dx), dy=int(dy), ncc=round(float(mx), 3), ncc0=round(float(z), 3), mad=round(mad, 2))
        out.append(res)

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for m in ["pixel", "cv"]:
    fp = [o for o in out if o["method"] == m and not o["tp"]]
    tp = [o for o in out if o["method"] == m and o["tp"]]
    print(m, "FP", len(fp), Counter(o["cls"] for o in fp), "| TP", len(tp), Counter(o["cls"] for o in tp))
