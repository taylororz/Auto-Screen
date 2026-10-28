#!/usr/bin/env python3
"""Rechteck um einen eingebrachten Fehler ziehen und Koordinaten ausgeben.
Aufruf: python3 bbox_picker.py runs/B1/screenshots/baseline_DE_DE/language_button.png [runs/A1/screenshots/baseline_DE_DE/language_button.png]
Mit zweitem Bild wird links die Referenz (A1) und rechts der Defekt (B1) gezeigt.
Gezogen wird nur auf der rechten Seite, ausgegeben werden Koordinaten in Pixeln des B1-Originalbilds.
Maus ziehen, ENTER bestätigen. Ausgabe: [x, y, w, h]."""
import sys, cv2, numpy as np

actual = cv2.imread(sys.argv[1])
h, w = actual.shape[:2]
if len(sys.argv) > 2:
    baseline = cv2.imread(sys.argv[2])
    if baseline.shape[:2] != (h, w):
        sys.exit(f"Größen unterschiedlich: {baseline.shape[:2]} vs {(h, w)}")
    gap = np.full((h, 10, 3), 255, np.uint8)
    img = np.hstack([baseline, gap, actual])
    offset = w + 10
else:
    img = actual
    offset = 0

s = 900 / h
small = cv2.resize(img, (int(img.shape[1] * s), int(h * s)))
if offset:
    cv2.putText(small, "A1 (Referenz)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(small, "B1 (Defekt)", (int((offset + 10) * s), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

x, y, bw, bh = cv2.selectROI("Fehlerbereich markieren (rechts), ENTER = OK", small, showCrosshair=True)
cv2.destroyAllWindows()
if bw == 0 or bh == 0:
    print("Keine Auswahl (kein Rechteck gezogen)")
    sys.exit(0)
if offset and x < int(offset * s):
    sys.exit("Rechteck liegt auf der Referenzseite (links). Bitte rechts ziehen.")
x -= int(offset * s)
print([int((x) / s), int(y / s), int(bw / s), int(bh / s)])
