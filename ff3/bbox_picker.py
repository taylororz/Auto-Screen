#!/usr/bin/env python3
"""Interactive ROI selector to draw a defect bounding box [x, y, w, h] on a screenshot."""
import sys, cv2, numpy as np

actual = cv2.imread(sys.argv[1])
h, w = actual.shape[:2]
if len(sys.argv) > 2:
    baseline = cv2.imread(sys.argv[2])
    if baseline.shape[:2] != (h, w):
        sys.exit(f"Dimensions differ: {baseline.shape[:2]} vs {(h, w)}")
    gap = np.full((h, 10, 3), 255, np.uint8)
    img = np.hstack([baseline, gap, actual])
    offset = w + 10
else:
    img = actual
    offset = 0

s = 900 / h
small = cv2.resize(img, (int(img.shape[1] * s), int(h * s)))
if offset:
    cv2.putText(small, "A1 (Reference)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(small, "B1 (Defect)", (int((offset + 10) * s), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

x, y, bw, bh = cv2.selectROI("Select defect region (right side), ENTER = OK", small, showCrosshair=True)
cv2.destroyAllWindows()
if bw == 0 or bh == 0:
    print("No selection (no rectangle drawn)")
    sys.exit(0)
if offset and x < int(offset * s):
    sys.exit("Rectangle is on the reference side (left). Please draw on the right side.")
x -= int(offset * s)
print([int((x) / s), int(y / s), int(bw / s), int(bh / s)])
