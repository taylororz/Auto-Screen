#!/usr/bin/env python3
"""OpenCV, Apple Vision OCR, and GPT-4o-mini UI grounding pipeline for iOS-1K."""

import os
import sys
import json
import shutil
import random
import argparse
import base64
import subprocess
from pathlib import Path
import cv2
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    for cand_env in [
        Path(__file__).resolve().parent / ".env",
        Path("/Users/taylorzhuang/social-app/Auto-Screen/.env"),
        Path.home() / ".env"
    ]:
        if cand_env.exists():
            load_dotenv(dotenv_path=cand_env)
except ImportError:
    pass

# Fallback direct .env parsing if dotenv is not installed
if not os.environ.get("OPENAI_API_KEY"):
    for cand_env in [
        Path(__file__).resolve().parent / ".env",
        Path("/Users/taylorzhuang/social-app/Auto-Screen/.env"),
        Path.home() / ".env"
    ]:
        if cand_env.exists():
            with open(cand_env) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip()

ROOT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("IOS1K_DIR", "/Users/taylorzhuang/screenshot/iOS-1K-Mobile-UI-Dataset"))
DIFF_DIR = ROOT_DIR / "diff_results" / "dataset_ios1k"
SOM_DIR = DIFF_DIR / "som_images"
FINETUNE_FILE = DIFF_DIR / "finetune_dataset.jsonl"

SWIFT_OCR_PATH = ROOT_DIR / "swift_ocr.swift"


# ==============================================================================
# 1. Computer Vision (OpenCV) Feature Extraction
# ==============================================================================
def _compute_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_contained_or_overlapping(b1, b2, threshold=0.65):
    """Checks if b1 is mostly contained in b2 or vice-versa."""
    ax, ay, aw, ah = b1["x"], b1["y"], b1.get("w", b1.get("width", 0)), b1.get("h", b1.get("height", 0))
    bx, by, bw, bh = b2["x"], b2["y"], b2.get("w", b2.get("width", 0)), b2.get("h", b2.get("height", 0))
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return False
    min_area = min(aw * ah, bw * bh)
    return (inter / min_area) >= threshold if min_area > 0 else False


def cv_detect_ui_regions(image_path, img_w=828, img_h=1792, status_bar_h=95, bottom_bar_h=65):
    """
    Extracts candidate UI bounding boxes using OpenCV edge, thresholding, and morphological contours.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candidate_boxes = []
    max_area = w_img * h_img * 0.38
    min_dim = 20

    # 1. Circular & shape button detection (Play buttons, Add (+), Avatar, More)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    cnts_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_thresh:
        x, y, w, h = cv2.boundingRect(c)
        if y < status_bar_h or y + h > h_img - bottom_bar_h:
            continue
        if 32 <= w <= 140 and 32 <= h <= 140 and 0.70 <= (w / h) <= 1.40:
            candidate_boxes.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "is_circle": True})

    # 2. Adaptive edge & morphological contours
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 140)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    kernel_box = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    morph = cv2.dilate(edges, kernel_h, iterations=1)
    morph = cv2.dilate(morph, kernel_v, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_box, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore status bar and home indicator regions
        if y + h <= status_bar_h or (y < status_bar_h and h < 60):
            continue
        if y >= h_img - bottom_bar_h or (y + h >= h_img - 25 and h < 50):
            continue

        # Ignore tiny noise or massive full-screen background wraps
        if w < min_dim or h < min_dim or (w * h > max_area):
            continue

        # Ignore screen-wide empty banner boxes
        if w > 0.85 * w_img and h > 0.30 * h_img:
            continue

        candidate_boxes.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "is_circle": False})

    # Detect iOS green active switches via HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    cnts_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_g:
        x, y, w, h = cv2.boundingRect(c)
        if x > 650 and 70 <= w <= 130 and 40 <= h <= 85:
            candidate_boxes.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "is_circle": False})

    # Prioritize specialized controls (e.g. green active switches) so they are never suppressed by noise fragments
    switch_boxes = [b for b in candidate_boxes if b["x"] > 650 and 70 <= b["w"] <= 130 and 40 <= b["h"] <= 85]
    other_boxes = [b for b in candidate_boxes if b not in switch_boxes]

    # Deduplicate: switches always take precedence
    deduped = list(switch_boxes)
    for box in other_boxes:
        if any(_is_contained_or_overlapping(box, s, threshold=0.50) for s in switch_boxes):
            continue
        if not any(_is_contained_or_overlapping(box, kept, threshold=0.75) for kept in deduped):
            deduped.append(box)

    return deduped


# ==============================================================================
# 2. Native Apple Vision OCR Integration
# ==============================================================================
def extract_vision_ocr(image_path):
    """
    Executes native Apple Vision OCR via swift helper, providing pixel-accurate multilingual text bboxes.
    """
    if not SWIFT_OCR_PATH.exists():
        return []

    try:
        res = subprocess.run(
            ["swift", str(SWIFT_OCR_PATH), str(image_path)],
            capture_output=True,
            text=True,
            timeout=15
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            # Filter status bar text (y < 85)
            return [it for it in data if it.get("y", 0) >= 85 and it.get("text", "").strip()]
    except Exception as e:
        print(f"⚠️ Vision OCR exception on {image_path}: {e}")

    return []


# ==============================================================================
# 3. Multimodal Spatial Fusion (Tab Bar, Rows, and Icons)
# ==============================================================================
def fuse_cv_and_ocr(cv_boxes, ocr_items, img_w=828, img_h=1792):
    """
    Fuses OpenCV geometric contours with Vision OCR text boxes.
    Produces tight, enriched candidate interactive controls:
    - Clusters bottom tab icons and text into unified Tab targets (y > 1580)
    - Envelopes list rows (thumbnails + titles + subtitles) into interactive cards
    - Standardizes navigation bar tap targets (back buttons, close buttons, action icons)
    """
    fused_elements = []
    used_ocr = set()
    used_cv = set()

    # A. Dynamic Bottom Navigation Bar or Wide Action Button (y >= 1540)
    bottom_ocrs = [(idx, it) for idx, it in enumerate(ocr_items) if it["y"] >= 1540]
    bottom_cvs = [(idx, b) for idx, b in enumerate(cv_boxes) if b["y"] >= 1520]

    # Check if there is a primary wide CTA button (e.g. "Continue", "Get Started", "UNLOCK WITH VIP PASS")
    wide_cta = None
    for idx, it in bottom_ocrs:
        txt = it["text"].strip()
        low_t = txt.lower()
        if it["w"] > 400 or low_t in ["continue", "get started", "next", "save", "submit", "done", "join", "sign in", "unlock with vip pass"]:
            wide_cta = (idx, it)
            break

    if wide_cta:
        w_idx, w_ocr = wide_cta
        used_ocr.add(w_idx)
        # Find matching wide CV button contour
        matched_btn = None
        for c_idx, b in bottom_cvs:
            if b["w"] > 350 and abs((b["x"] + b["w"]/2.0) - (w_ocr["x"] + w_ocr["w"]/2.0)) < 80:
                matched_btn = b
                used_cv.add(c_idx)
                break
        bx = matched_btn["x"] if matched_btn else max(30.0, w_ocr["x"] - 25.0)
        by = matched_btn["y"] if matched_btn else max(1540.0, w_ocr["y"] - 15.0)
        bw = matched_btn["w"] if matched_btn else min(float(img_w) - 60.0, w_ocr["w"] + 50.0)
        bh = matched_btn["h"] if matched_btn else max(w_ocr["h"] + 30.0, 85.0)
        fused_elements.append({
            "bbox": {"x": round(bx, 1), "y": round(by, 1), "w": round(bw, 1), "h": round(bh, 1)},
            "text": w_ocr["text"].strip(),
            "type_hint": "Button",
            "has_text": True,
            "is_icon": False
        })
    else:
        # Check if bottom items are a multi-tab bar (e.g. 3 to 5 tabs at bottom)
        tab_candidates = []
        for idx, it in bottom_ocrs:
            txt = it["text"].strip()
            # Valid tab text: short string, width < 260
            if it["w"] < 260 and len(txt) <= 22 and it["y"] >= 1600:
                used_ocr.add(idx)
                cx = it["x"] + it["w"] / 2.0
                tab_candidates.append({
                    "text": txt,
                    "cx": cx,
                    "min_x": it["x"],
                    "max_x": it["x"] + it["w"]
                })
        # Absorb bottom CV icon boxes into tabs
        for c_idx, b in bottom_cvs:
            if b["y"] >= 1580 and b["w"] <= 140 and b["h"] <= 140:
                used_cv.add(c_idx)
                cx = b["x"] + b["w"] / 2.0
                matched_tab = False
                for tc in tab_candidates:
                    if abs(tc["cx"] - cx) < 65:
                        tc["min_x"] = min(tc["min_x"], b["x"])
                        tc["max_x"] = max(tc["max_x"], b["x"] + b["w"])
                        matched_tab = True
                        break
                if not matched_tab and len(tab_candidates) < 5:
                    tab_candidates.append({
                        "text": "",
                        "cx": cx,
                        "min_x": b["x"],
                        "max_x": b["x"] + b["w"]
                    })
        # If genuine tab bar detected (2 or more tabs)
        if len(tab_candidates) >= 2:
            tab_candidates.sort(key=lambda t: t["cx"])
            for tc in tab_candidates:
                bw = max(tc["max_x"] - tc["min_x"] + 30.0, 110.0)
                bx = max(10.0, tc["cx"] - bw / 2.0)
                if bx + bw > img_w - 10.0:
                    bx = img_w - 10.0 - bw
                fused_elements.append({
                    "bbox": {"x": round(bx, 1), "y": 1618.0, "w": round(bw, 1), "h": 110.0},
                    "text": tc["text"],
                    "type_hint": "Tab",
                    "has_text": bool(tc["text"]),
                    "is_icon": not bool(tc["text"])
                })

    # Detect Split-View layout (e.g. Left vertical category rail at x < 210)
    left_rail_ocrs = [
        (idx, it) for idx, it in enumerate(ocr_items)
        if idx not in used_ocr and it["x"] < 210 and 200 <= it["y"] < 1350 and 3 <= len(it["text"].strip()) <= 25
    ]
    # A true split-view rail requires that right side also has content (e.g. at x > 250)
    has_right_content = any(it["x"] > 250 and 200 <= it["y"] < 1350 for it in ocr_items)
    has_split_view = (len(left_rail_ocrs) >= 4 and has_right_content)

    # If genuine split-view layout, process the ACTUAL detected left category rail items
    if has_split_view:
        for idx, it in left_rail_ocrs:
            used_ocr.add(idx)
            fused_elements.append({
                "bbox": {"x": 0.0, "y": round(it["y"] - 15.0, 1), "w": 205.0, "h": max(it["h"] + 30.0, 75.0)},
                "text": it["text"].strip(),
                "type_hint": "Row",
                "has_text": True,
                "is_icon": False
            })

    # B0. Filter top navigation bar static title (y < 180, 120 < x < 700)
    for o_idx, ocr in enumerate(ocr_items):
        if o_idx not in used_ocr and ocr["y"] < 180 and 120 < ocr["x"] < 700:
            if not has_split_view:  # Don't filter search bar in split view
                used_ocr.add(o_idx)

    # B. Navigation Bar Icons & Back Buttons (y < 220, x < 150)
    for o_idx, ocr in enumerate(ocr_items):
        if o_idx in used_ocr:
            continue
        if ocr["y"] < 220 and ocr["x"] < 150:
            low_t = ocr["text"].strip().lower()
            if low_t in ["<", "‹", "«", "←", "back", "x", "✕", "✖", "close", "done"]:
                bx = max(10.0, ocr["x"] - 14.0)
                by = max(95.0, ocr["y"] - 14.0)
                bw = max(ocr["w"] + 28.0, 68.0)
                bh = max(ocr["h"] + 28.0, 68.0)
                fused_elements.append({
                    "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
                    "text": ocr["text"].strip(),
                    "type_hint": "Icon",
                    "has_text": True,
                    "is_icon": True
                })
                used_ocr.add(o_idx)

    # B1. Navigation rows with right chevron ('>' or 'v' dropdown)
    min_row_x = 215.0 if has_split_view else 20.0
    max_title_x = 650.0
    for o_idx, ocr in enumerate(ocr_items):
        if o_idx not in used_ocr and ocr["x"] > 680 and ocr["text"].strip().lower() in [">", "›", "»", "→", "v", "⌵", "⌄"] and 150 <= ocr["y"] < 1550:
            used_ocr.add(o_idx)
            # Find row title to the left, respecting split view boundary
            row_ocrs = [
                (i, o) for i, o in enumerate(ocr_items)
                if i not in used_ocr and (min_row_x <= o["x"] < max_title_x) and abs(o["y"] - ocr["y"]) < 50
            ]
            row_ocrs.sort(key=lambda t: t[1]["y"])
            if row_ocrs:
                best_i, best_o = row_ocrs[0]
                for i_sub, _ in row_ocrs:
                    used_ocr.add(i_sub)
                fused_elements.append({
                    "bbox": {"x": min_row_x, "y": round(best_o["y"] - 14.0, 1), "w": float(img_w) - min_row_x - 20.0, "h": max(ocr["y"] - best_o["y"] + 65.0, 80.0)},
                    "text": best_o["text"].strip(),
                    "type_hint": "Row",
                    "has_text": True,
                    "is_icon": False
                })

    # B2. Native iOS Switch Knob Detection (Requires real capsule dimensions 80-130px)
    switch_candidates = []
    for c_idx, b in enumerate(cv_boxes):
        if c_idx not in used_cv and b["x"] > 670 and 150 <= b["y"] < 1550:
            is_capsule = (80 <= b["w"] <= 130 and 45 <= b["h"] <= 85)
            if is_capsule and not any(abs(b["y"] - kept["y"]) < 30 for kept in switch_candidates):
                switch_candidates.append(b)
                used_cv.add(c_idx)

    for b in switch_candidates:
        # Find nearest title above or level with this switch (y in [b["y"] - 65, b["y"] + 40])
        cand_ocrs = [(i, o) for i, o in enumerate(ocr_items) if i not in used_ocr and o["x"] < 650 and b["y"] - 65 <= o["y"] <= b["y"] + 40]
        cand_ocrs.sort(key=lambda t: t[1]["y"])
        if cand_ocrs:
            best_i, best_o = cand_ocrs[0]
            # Consume all subtitle lines in this switch row block to prevent them becoming stray buttons
            for idx, o in enumerate(ocr_items):
                if idx not in used_ocr and o["x"] < 650 and b["y"] - 65 <= o["y"] <= b["y"] + 90:
                    used_ocr.add(idx)
            sw_cy = b["y"] + b["h"] / 2.0
            fused_elements.append({
                "bbox": {"x": 697.0, "y": round(sw_cy - 32.0, 1), "w": 104.0, "h": 64.0},
                "text": best_o["text"].strip(),
                "type_hint": "Switch",
                "has_text": True,
                "is_icon": False
            })

    # B3. Native Action Pill Buttons on right edge (Remove, Clear, Connect, Get app, Follow)
    for o_idx, ocr in enumerate(ocr_items):
        if o_idx not in used_ocr and ocr["x"] > 600 and 150 <= ocr["y"] < 1550:
            txt = ocr["text"].strip()
            if txt in ["Remove", "Clear", "Connect", "Get app", "Follow", "Install", "Edit", "Update"]:
                matched_cv = None
                for c_idx, b in enumerate(cv_boxes):
                    if c_idx not in used_cv and abs(b["x"] - ocr["x"]) < 50 and abs(b["y"] - ocr["y"]) < 30:
                        matched_cv = b
                        used_cv.add(c_idx)
                        break
                bx = max(620.0, (matched_cv["x"] - 15.0) if matched_cv else (ocr["x"] - 18.0))
                by = (matched_cv["y"] - 10.0) if matched_cv else (ocr["y"] - 12.0)
                bw = (matched_cv["w"] + 30.0) if matched_cv else max(ocr["w"] + 36.0, 120.0)
                bh = (matched_cv["h"] + 20.0) if matched_cv else (ocr["h"] + 24.0)
                used_ocr.add(o_idx)
                fused_elements.append({
                    "bbox": {"x": round(bx, 1), "y": round(by, 1), "w": round(bw, 1), "h": round(bh, 1)},
                    "text": txt,
                    "type_hint": "Button",
                    "has_text": True,
                    "is_icon": False
                })

    # C. Content List Rows / Cards / Buttons Enveloping
    content_ocrs = [
        (idx, it) for idx, it in enumerate(ocr_items)
        if idx not in used_ocr and 180 <= it["y"] < 1580
    ]
    content_ocrs.sort(key=lambda t: t[1]["y"])

    # Cluster OCR items vertically into rows (same column alignment & small vertical gap)
    row_clusters = []
    for idx, ocr in content_ocrs:
        added = False
        for rc in row_clusters:
            last_ocr = rc[-1][1]
            if abs(ocr["x"] - last_ocr["x"]) < 50 and 0 <= (ocr["y"] - (last_ocr["y"] + last_ocr["h"])) < 35:
                rc.append((idx, ocr))
                added = True
                break
        if not added:
            row_clusters.append([(idx, ocr)])

    # Compute bounds and check for side-by-side elements (e.g. filter pills)
    cluster_info = []
    for rc in row_clusters:
        c_items = [t[1] for t in rc]
        min_x = min(it["x"] for it in c_items)
        max_x = max(it["x"] + it["w"] for it in c_items)
        min_y = min(it["y"] for it in c_items)
        max_y = max(it["y"] + it["h"] for it in c_items)
        cluster_info.append({
            "rc": rc,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "ymid": (min_y + max_y) / 2.0
        })

    for i, ci in enumerate(cluster_info):
        rc = ci["rc"]
        min_x, max_x, min_y, max_y = ci["min_x"], ci["max_x"], ci["min_y"], ci["max_y"]
        c_indices = [t[0] for t in rc]
        c_items = [t[1] for t in rc]
        cx = (min_x + max_x) / 2.0
        # Skip known static section headers and static info rows that have no interaction
        comb_text_low = " ".join(it["text"].lower() for it in c_items).strip()
        is_header = any(comb_text_low.endswith(s) or comb_text_low == s for s in [
            "activity", "visibility", "preferences", "settings", "options", 
            "streaming", "storage", "storage breakdown", "information", "social", "devices", "in the car", "connected apps",
            "version"
        ]) or comb_text_low.startswith("version ") or comb_text_low.replace(".", "").isdigit()
        if is_header:
            for idx in c_indices:
                used_ocr.add(idx)
            continue

        # Check if side-by-side with another element on similar y (pill buttons/segmented controls)
        is_sbs = any(j != i and abs(other["ymid"] - ci["ymid"]) < 25 for j, other in enumerate(cluster_info))
        is_centered = abs(cx - img_w / 2.0) < 60 and (max_x - min_x) < 400

        # Match CV thumbnail to the left (a real thumbnail must be at x >= 20, min_x >= 120, w >= 70, h >= 70)
        matched_thumb = None
        if min_x >= 120:
            for c_idx, b in enumerate(cv_boxes):
                if c_idx not in used_cv and b["y"] >= 180 and b["x"] < min_x - 30 and b["w"] >= 70 and b["h"] >= 70:
                    if max(min_y - 20, b["y"]) < min(max_y + 20, b["y"] + b["h"]):
                        matched_thumb = b
                        used_cv.add(c_idx)
                        break

        has_leading_thumb = (110 <= min_x <= 250) and (matched_thumb is not None)

        first_txt = c_items[0]["text"].strip()
        is_option = first_txt in ["Automatic", "Low", "Normal", "High", "Very high", "Standard", "Extreme"]
        base_left_x = 215.0 if (has_split_view and min_x >= 180) else 20.0
        base_row_w = (float(img_w) - base_left_x - 20.0) if (has_split_view and min_x >= 180) else (float(img_w) - 40.0)

        if is_option:
            # Full width option row
            row_x = base_left_x
            row_w = base_row_w
            row_y = min_y - 12.0
            row_h = max((max_y - min_y) + 24.0, 52.0)
            t_hint = "Row"
            # Absorb any trailing badge on the right (e.g. • Premium)
            for other_ci in cluster_info:
                if abs(other_ci["ymid"] - ci["ymid"]) < 25 and other_ci["min_x"] > 500:
                    for other_idx in [t[0] for t in other_ci["rc"]]:
                        used_ocr.add(other_idx)
        elif is_sbs or (max_x - min_x < 280 and min_x > 50 and not has_leading_thumb):
            # Pill button / chip (e.g. Music, Podcasts & Shows)
            row_x = max(base_left_x, min_x - 14.0)
            row_w = min(float(img_w) - row_x, (max_x - min_x) + 28.0)
            row_y = max(180.0, min_y - 10.0)
            row_h = (max_y - min_y) + 20.0
            t_hint = "Button"
        elif is_centered:
            # Centered button (e.g. Log out)
            row_x = max(base_left_x, min_x - 50.0)
            row_w = min(float(img_w) - row_x, (max_x - min_x) + 100.0)
            row_y = min_y - 12.0
            row_h = (max_y - min_y) + 24.0
            t_hint = "Button"
        elif has_leading_thumb:
            # Full row card with avatar / cover art thumbnail
            row_x = base_left_x
            row_w = base_row_w
            row_y = min_y - 14.0
            if matched_thumb:
                row_y = min(row_y, matched_thumb["y"] - 6.0)
            row_h = max(max_y - row_y + 14.0, 105.0)
            t_hint = "Card"
        else:
            # Standard settings / menu row (no thumbnail, clean thin row to prevent overlap)
            row_x = base_left_x
            row_w = base_row_w
            row_y = min_y - 10.0
            row_h = (max_y - min_y) + 22.0
            t_hint = "Row"

        for idx in c_indices:
            used_ocr.add(idx)

        comb_text = " ".join(it["text"] for it in c_items)
        fused_elements.append({
            "bbox": {"x": round(row_x, 1), "y": round(row_y, 1), "w": round(row_w, 1), "h": round(row_h, 1)},
            "text": comb_text,
            "type_hint": t_hint,
            "has_text": True,
            "is_icon": False
        })

    # D. Remaining CV contours & OCR items
    for idx, box in enumerate(cv_boxes):
        if idx in used_cv:
            continue
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        matched_texts = []
        for o_idx, ocr in enumerate(ocr_items):
            if o_idx in used_ocr:
                continue
            ox, oy, ow, oh = ocr["x"], ocr["y"], ocr["w"], ocr["h"]
            cx, cy = ox + ow / 2, oy + oh / 2
            if (bx - 10 <= cx <= bx + bw + 10) and (by - 10 <= cy <= by + bh + 10):
                matched_texts.append(ocr["text"].strip())
                used_ocr.add(o_idx)

        # Skip empty CV contours that overlap static text already filtered in used_ocr (e.g. titles, version)
        if not matched_texts:
            overlaps_used_ocr = any(
                (bx - 10 <= ocr_items[u]["x"] + ocr_items[u]["w"] / 2 <= bx + bw + 10) and
                (by - 10 <= ocr_items[u]["y"] + ocr_items[u]["h"] / 2 <= by + bh + 10)
                for u in used_ocr
            )
            if overlaps_used_ocr:
                continue

        comb_text = " ".join(matched_texts) if matched_texts else ""
        if not comb_text and (bw > 110 or bh > 110):
            # Purely decorative graphic or empty container, skip
            continue

        fused_elements.append({
            "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
            "text": comb_text,
            "type_hint": "Icon" if not comb_text else ("Row" if bw > 450 else "Button"),
            "has_text": bool(comb_text),
            "is_icon": not bool(comb_text)
        })

    # Add any remaining unused OCR items
    for o_idx, ocr in enumerate(ocr_items):
        if o_idx not in used_ocr:
            ox, oy, ow, oh = ocr["x"], ocr["y"], ocr["w"], ocr["h"]
            if ow >= 14 and oh >= 10:
                fused_elements.append({
                    "bbox": {"x": max(0.0, ox - 10.0), "y": max(0.0, oy - 8.0), "w": ow + 20.0, "h": oh + 16.0},
                    "text": ocr["text"].strip(),
                    "has_text": True,
                    "is_icon": False
                })

    # Spatial deduplication
    fused_elements.sort(key=lambda e: (e["bbox"]["y"] // 40, e["bbox"]["x"]))
    final_cands = []
    for el in fused_elements:
        if not any(_is_contained_or_overlapping(el["bbox"], kept["bbox"], threshold=0.70) for kept in final_cands):
            final_cands.append(el)

    return final_cands


def clean_instruction(el_type, text, raw_inst, item_idx=None, total_items=None):
    """Normalizes instruction text to standard dataset conventions."""
    txt = (text or "").strip()
    low_t = txt.lower()
    raw = (raw_inst or "").strip()
    low_raw = raw.lower()

    if "search" in low_t:
        return "Click the search bar"

    if low_t in ["<", "‹", "«", "←", "back"] or ("back button" in low_raw and (not txt or low_t in ["<", "‹", "«", "←", "back"])):
        return "Click the back button"

    if el_type == "Input":
        return "Click the search bar"

    if el_type == "Tab":
        tab_name = txt if txt else "Home"
        return f"Click the '{tab_name}' tab"

    if "play button" in low_raw or low_t in ["play", "▶"]:
        if "first" in low_raw:
            return "Click the first play button"
        elif "second" in low_raw:
            return "Click the second play button"
        return "Click the play button"

    if "add button" in low_raw or low_t in ["+", "add"]:
        if "first" in low_raw:
            return "Click the first add button"
        elif "second" in low_raw:
            return "Click the second add button"
        return "Click the add button"

    if "more options" in low_raw or low_t in ["•••", "...", "more"]:
        if "first" in low_raw:
            return "Click the '•••' more options button for the first release"
        elif "second" in low_raw:
            return "Click the '•••' more options button for the second release"
    if el_type == "Switch" or "toggle" in low_raw or "switch" in low_raw:
        clean_txt = txt.split("\n")[0].strip()
        return f"Click the '{clean_txt}' toggle"

    # Check for release wording: ONLY allow 'release' if text actually contains music release indicators
    if "first release" in low_raw or "second release" in low_raw or "release" in low_raw:
        if not any(w in low_t for w in ["single", "album", "ep", "latest release", "song", "track"]):
            clean_txt = txt.split("\n")[0].strip()
            if "profile" in low_t:
                return "Click the 'View Profile' row"
            return f"Click the '{clean_txt}' row"

    # If raw_inst is already a well-formed "Click the ...", validate it
    if raw.startswith("Click the ") or raw.startswith("Click '") or raw.startswith("Click "):
        # Discard broken generic phrases
        if not any(bad in low_raw for bad in ["'add' button", "'single' item", "'single' button", "'•••' button", "release '"]):
            return raw

    if el_type in ["Card", "Row"]:
        clean_txt = txt.split("\n")[0].strip()
        clean_txt = clean_txt.replace("Single", "").replace("•", "").strip()
        if "view profile" in low_t or "profile" in low_t:
            return "Click the 'View Profile' row"
        return f"Click the '{clean_txt}' row"

    if el_type == "Input":
        return f"Click the '{txt}' input field"

    if txt:
        return f"Click the '{txt}' button"

    return raw or "Click the button"


def resolve_vertical_overlaps(elements):
    """
    Prevents any bounding box collision / overlapping between list rows or cards.
    Ensures minimum 4px separation between all full-width items.
    """
    # Find full-width rows and cards
    full_rows = [el for el in elements if el.get("bbox", {}).get("w", 0) > 400 and el.get("bbox", {}).get("y", 0) < 1550]
    full_rows.sort(key=lambda el: el["bbox"]["y"])

    for i in range(len(full_rows) - 1):
        r1 = full_rows[i]["bbox"]
        r2 = full_rows[i+1]["bbox"]
        y1_bot = r1["y"] + r1["h"]
        if y1_bot > r2["y"]:
            overlap = y1_bot - r2["y"]
            adj = (overlap + 6.0) / 2.0
            r1["h"] = round(max(32.0, r1["h"] - adj), 1)
            r2["y"] = round(r1["y"] + r1["h"] + 4.0, 1)
            r2["h"] = round(max(32.0, r2["h"] - adj), 1)

    # Recompute centers
    for el in elements:
        b = el["bbox"]
        el["center"] = {"x": int(b["x"] + b["w"] / 2), "y": int(b["y"] + b["h"] / 2)}

    return elements


# ==============================================================================
# 4. AI Semantic Grounding with Token Optimization & Exponential Retry
# ==============================================================================
def ai_semantic_grounding(image_path, candidates, model="gpt-4o-mini"):
    """
    Calls OpenAI GPT-4o-mini vision to inspect candidates, eliminate background/occluded items,
    classify interactive roles, and synthesize standard task instructions.
    Uses resized JPEG encoding (saves 65% tokens) and exponential backoff retry on 429.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_local_grounding(candidates)

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        return _fallback_local_grounding(candidates)

    h_orig, w_orig = img.shape[:2]

    # Downsample image to max dim 1024 for lightweight VLM prompt (prevents 429 rate limit)
    scale = 1024.0 / max(h_orig, w_orig)
    resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buf).decode("utf-8")

    prompt = (
        "You are an expert Mobile UI Testing & Annotation Agent for iOS applications.\n"
        "Attached is an iOS screen image and a list of candidate bounding box detections with OCR text.\n\n"
        "Your task:\n"
        "1. Identify ALL ACTIVE, CLICKABLE / INTERACTIVE UI controls visible on this screen.\n"
        "2. CRITICAL RULES:\n"
        "   - Top navigation back button '<' or 'X' must ALWAYS be kept if visible on top-left!\n"
        "   - Page title text centered in navigation bar (e.g. 'Playlists', 'Settings', 'LANGUAGE', 'About', 'Profile') is a STATIC TITLE, NOT a button. Do NOT mark titles.\n"
        "   - Do NOT mark purely decorative logos, illustrations, or empty image containers (e.g. center icons without click action).\n"
        "   - Do NOT mark static informational paragraphs, explanations, quotes, or verses (e.g. 'In the Name of Allah...', 'Understand Quran in your language...').\n"
        "   - Action buttons: 'Continue', 'Get Started', 'UNLOCK WITH VIP PASS', 'Sync Now', 'Create', 'Save', etc. are PRIMARY CLICKABLE BUTTONS. ALWAYS mark them!\n"
        "   - Selectable list rows / options (e.g. language selection list 'English', 'Bahasa Melayu', 'Bosanski', etc.): mark each selectable row as 'Row'!\n"
        "   - Settings / Form rows with '>' or dropdown: mark as 'Row' (e.g. 'Click the '<Name>' row').\n"
        "   - Filter chips / pills (e.g. 'Collections', 'Pins', 'Notes', 'Male', 'Female'): mark as 'Button'.\n"
        "   - Bottom Tab bar items (e.g. Home, Planner, Library, Stats): mark each tab with its actual visible label as 'Tab'!\n"
        "   - Inputs / Search bars: mark as 'Input'.\n"
        "3. ACCURATE INSTRUCTION SYNTAX:\n"
        "   Every instruction MUST start with 'Click the ...' following these exact conventions:\n"
        "   - Back button: 'Click the back button'\n"
        "   - Button: 'Click the '<Name>' button' (e.g. 'Click the 'Continue' button', 'Click the 'Get Started' button')\n"
        "   - Row / List Option: 'Click the '<Name>' row' (e.g. 'Click the 'English' row', 'Click the 'Bahasa Melayu' row')\n"
        "   - Tab: 'Click the '<Name>' tab' (e.g. 'Click the 'Home' tab', 'Click the 'Library' tab')\n"
        "   - Search / Input: 'Click the search bar' or 'Click the '<Name>' input field'\n"
        "   - Switch: 'Click the '<Name>' toggle'\n"
        "4. Output ONLY valid JSON selecting interactive elements from the Candidate Boxes. Specify \"cand_id\":\n"
        "{\"elements\": [\n"
        "  {\"cand_id\": 1, \"type\": \"Icon\", \"instruction\": \"Click the back button\"},\n"
        "  {\"cand_id\": 3, \"type\": \"Button\", \"instruction\": \"Click the 'Continue' button\"}\n"
        "]}\n"
    )

    cands_summary = []
    for idx, c in enumerate(candidates, 1):
        b = c["bbox"]
        cands_summary.append({
            "cand_id": idx,
            "type_hint": c.get("type_hint", "Button"),
            "bbox": [round(b["x"], 1), round(b["y"], 1), round(b["w"], 1), round(b["h"], 1)],
            "ocr_text": c["text"]
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + f"\nCandidate Boxes (in 828x1792):\n{json.dumps(cands_summary[:45], ensure_ascii=False)}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2048
    }

    # Exponential retry loop for rate limit resilience
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=35
            )
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(result)
                elements = parsed.get("elements", [])
                formatted = []
                for tid, el in enumerate(elements, 1):
                    cand_id = el.get("cand_id")
                    if cand_id is not None and isinstance(cand_id, int) and 1 <= cand_id <= len(candidates):
                        cand_box = candidates[cand_id - 1]["bbox"]
                        bx, by, bw, bh = float(cand_box["x"]), float(cand_box["y"]), float(cand_box["w"]), float(cand_box["h"])
                        if not el.get("text"):
                            el["text"] = candidates[cand_id - 1]["text"]
                    else:
                        b = el.get("bbox", [0, 0, 0, 0])
                        if isinstance(b, list) and len(b) >= 4:
                            bx, by, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        elif isinstance(b, dict):
                            bx, by = float(b.get("x", 0)), float(b.get("y", 0))
                            bw = float(b.get("w", b.get("width", 0)))
                            bh = float(b.get("h", b.get("height", 0)))
                        else:
                            continue

                    if bw < 16 or bh < 16:
                        continue

                    el_type = el.get("type", "Button")
                    txt = el.get("text", "")
                    low_t = txt.lower()

                    # Respect the fused candidate coordinates if cand_id was used
                    is_cand = (cand_id and 1 <= cand_id <= len(candidates))
                    
                    if not is_cand:
                        if el_type == "Tab" or by >= 1580:
                            by = 1618.0
                            bh = 110.0
                            bw = max(bw + 30.0, 108.0)
                            bx = max(15.0, bx - 15.0)
                            el_type = "Tab"
                        elif low_t in ["<", "‹", "«", "←", "back", "close"]:
                            bx = max(10.0, bx - 14.0)
                            by = max(95.0, by - 14.0)
                            bw = max(bw + 28.0, 68.0)
                            bh = max(bh + 28.0, 68.0)
                            el_type = "Icon"
                        elif el_type == "Switch" or "toggle" in el.get("instruction", "").lower():
                            bx = max(660.0, bx - 10.0)
                            by = max(180.0, by - 8.0)
                            bw = 108.0
                            bh = 72.0
                            el_type = "Switch"
                        elif el_type in ["Card", "Row"] and (bw > 250):
                            bx = 20.0
                            bw = float(w_orig) - 40.0
                            by = max(180.0, by - 6.0)
                            bh = min(bh + 14.0, 78.0) if bh < 75.0 else bh
                        elif el_type == "Button" and bw < 300:
                            bx = max(15.0, bx - 12.0)
                            by = max(180.0, by - 10.0)
                            bw = bw + 24.0
                            bh = bh + 20.0
                    else:
                        cand_type = candidates[cand_id - 1].get("type_hint")
                        if cand_type and el_type in ["Button", "element", "Card"]:
                            el_type = cand_type

                    # Do NOT filter out prominent wide action buttons at the bottom
                    if by > 1540.0 and el_type not in ["Tab", "Button"] and bw < 300.0:
                        continue

                    inst_text = clean_instruction(el_type, txt, el.get("instruction", ""))

                    formatted.append({
                        "target_id": tid,
                        "type": el_type,
                        "text": txt,
                        "instruction": inst_text,
                        "isClickable": True,
                        "bbox": {"x": round(bx, 1), "y": round(by, 1), "w": round(bw, 1), "h": round(bh, 1)},
                        "center": {"x": int(bx + bw / 2), "y": int(by + bh / 2)}
                    })
                formatted = resolve_vertical_overlaps(formatted)
                return formatted
            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 2.5
                print(f"  ⏳ Rate limit 429 encountered. Backing off {wait_time:.1f}s (retry {attempt+1}/4)...")
                import time
                time.sleep(wait_time)
            else:
                print(f"⚠️ OpenAI Vision API error {resp.status_code}: {resp.text}")
                break
        except Exception as e:
            print(f"⚠️ OpenAI Vision API request exception (attempt {attempt+1}): {e}")
            import time
            time.sleep(2.0)

    return _fallback_local_grounding(candidates)


def _fallback_local_grounding(candidates):
    """Fallback rule-based semantic assigner if API is offline."""
    results = []
    tid = 1
    for c in candidates:
        b = c["bbox"]
        t = c["text"]
        low_t = t.lower()
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]

        # Classification heuristics
        if low_t in ["back", "<", "‹", "«", "←"]:
            el_type, inst = "Icon", "Click the back button"
        elif low_t in ["close", "done", "cancel", "x", "✕", "✖", "×"]:
            el_type, inst = "Icon", "Click the close button"
        elif "search" in low_t:
            el_type = "Input" if bw > 200 else "Icon"
            inst = "Click the search bar" if bw > 200 else "Tap the search icon"
        elif by > 1580 and bw <= 200:
            el_type, inst = "Tab", f"Click the '{t}' tab" if t else "Tap the navigation tab"
        elif c["is_icon"]:
            el_type, inst = "Icon", "Tap the action icon"
        elif t and len(t) <= 35:
            el_type, inst = "Button", f"Click the '{t}' button"
        else:
            continue

        results.append({
            "target_id": tid,
            "type": el_type,
            "text": t,
            "instruction": inst,
            "isClickable": True,
            "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
            "center": {"x": int(bx + bw / 2), "y": int(by + bh / 2)}
        })
        tid += 1

    return results


# ==============================================================================
# 5. Master Annotation & SoM Rendering
# ==============================================================================
def draw_som_image(image_path, elements, output_path):
    """
    Renders 100% target-aligned Set-of-Marks (SoM) bounding boxes with high-contrast numbered badges.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    annotated = image.copy()

    for el in elements:
        target_id = el["target_id"]
        bbox = el["bbox"]
        bx = int(bbox["x"])
        by = int(bbox["y"])
        bw = int(bbox.get("w", bbox.get("width", 0)))
        bh = int(bbox.get("h", bbox.get("height", 0)))
        label = f"[{target_id}]"

        box_color = (0, 0, 255)  # Bright Red
        badge_bg = (255, 255, 255)  # White Badge Background

        # Draw bounding box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

        # Calculate badge dimension
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        badge_x = max(2, bx)
        badge_y = max(th + 6, by)

        # Draw filled badge rectangle
        cv2.rectangle(
            annotated,
            (badge_x - 2, badge_y - th - 4),
            (badge_x + tw + 4, badge_y + baseline + 2),
            badge_bg,
            -1
        )
        cv2.rectangle(
            annotated,
            (badge_x - 2, badge_y - th - 4),
            (badge_x + tw + 4, badge_y + baseline + 2),
            box_color,
            1,
            cv2.LINE_AA
        )

        # Draw ID text
        cv2.putText(
            annotated,
            label,
            (badge_x + 1, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            box_color,
            thickness,
            cv2.LINE_AA
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return str(output_path)


def get_reviewed_stems():
    """Identifies all screen stems already reviewed by human in finetune_dataset.jsonl."""
    reviewed = set()
    if FINETUNE_FILE.exists():
        with open(FINETUNE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        sp = d.get("screenshot_path", "")
                        if sp:
                            reviewed.add(Path(sp).stem)
                    except Exception:
                        pass
    return reviewed


def extract_ui_elements(json_path, img_w=828, img_h=1792):
    """
    Backward-compatible loader. If json_path contains structured ground truth, returns elements directly.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return []

    elements = []
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], dict) and "elements" in data["annotations"]:
            elements = data["annotations"]["elements"]
        elif "elements" in data and isinstance(data["elements"], list):
            elements = data["elements"]

    cleaned = []
    for idx, el in enumerate(elements, 1):
        if not isinstance(el, dict):
            continue
        b = el.get("bbox") or el.get("bounds") or el.get("rect") or {}
        bx = float(b.get("x", 0))
        by = float(b.get("y", 0))
        bw = float(b.get("w", b.get("width", 0)))
        bh = float(b.get("h", b.get("height", 0)))

        cleaned.append({
            "target_id": el.get("target_id", idx),
            "type": el.get("type", "Button"),
            "text": el.get("text", ""),
            "instruction": el.get("instruction", f"Click the '{el.get('text', 'button')}' button"),
            "isClickable": el.get("isClickable", True),
            "bbox": {"x": bx, "y": by, "width": bw, "height": bh, "w": bw, "h": bh},
            "center": el.get("center", {"x": int(bx + bw / 2), "y": int(by + bh / 2)})
        })

    return cleaned


def annotate_screen_cv_ai(image_path, json_output_path=None, use_ai=True, model="gpt-4o-mini"):
    """
    Runs the complete OpenCV + Vision OCR + GPT-4o-mini pipeline on a single screen.
    Saves clean annotations to json_output_path if provided.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    # 1. OpenCV visual contour candidate detection
    cv_boxes = cv_detect_ui_regions(image_path, img_w, img_h)

    # 2. Apple Vision OCR text detection
    ocr_items = extract_vision_ocr(image_path)

    # 3. Multimodal Spatial Fusion
    candidates = fuse_cv_and_ocr(cv_boxes, ocr_items, img_w, img_h)

    # 4. AI Semantic Grounding & Refinement
    if use_ai:
        elements = ai_semantic_grounding(image_path, candidates, model=model)
    else:
        elements = _fallback_local_grounding(candidates)

    # 4.5 Sub-pixel Switch Physical Calibration
    # Detect all physical switch knobs and snap Switch boxes to standard UISwitch capsule
    gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_cv = cv2.threshold(gray_cv, 200, 255, cv2.THRESH_BINARY)
    cnts_cv, _ = cv2.findContours(thresh_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    knobs_detected = []
    for c in cnts_cv:
        kx, ky, kw, kh = cv2.boundingRect(c)
        if kx > 685 and 36 <= kw <= 65 and 36 <= kh <= 65 and 150 < ky < 1550:
            knobs_detected.append(ky + kh / 2.0)

    for el in elements:
        if el.get("type") == "Switch":
            ey = el["bbox"]["y"] + el["bbox"]["h"] / 2.0
            close_knobs = [kcy for kcy in knobs_detected if abs(kcy - ey) < 55]
            if close_knobs:
                best_kcy = min(close_knobs, key=lambda kcy: abs(kcy - ey))
                el["bbox"]["x"] = 697.0
                el["bbox"]["y"] = round(best_kcy - 32.0, 1)
                el["bbox"]["w"] = 104.0
                el["bbox"]["h"] = 64.0
                el["center"]["x"] = int(el["bbox"]["x"] + el["bbox"]["w"] / 2)
                el["center"]["y"] = int(el["bbox"]["y"] + el["bbox"]["h"] / 2)

    # 5. Save JSON annotation if requested
    if json_output_path:
        json_output_path = Path(json_output_path)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "screen_id": image_path.stem,
                "annotations": {
                    "elements": elements,
                    "count": len(elements)
                }
            }, f, indent=2, ensure_ascii=False)

    return elements


# ==============================================================================
# 6. Synchronizing Datasets (SOM & DG)
# ==============================================================================
def update_datasets_for_screen(stem, elements, img_path, som_img_path):
    """
    Synchronizes the screen records across all 4 training & benchmark files:
    1. ios1k_som_train.jsonl
    2. ios1k_dg_train.jsonl
    3. qwen_vl_ios1k_som_train.json
    4. qwen_vl_ios1k_dg_train.json
    """
    som_file = DIFF_DIR / "ios1k_som_train.jsonl"
    dg_file = DIFF_DIR / "ios1k_dg_train.jsonl"
    qwen_som_file = DIFF_DIR / "qwen_vl_ios1k_som_train.json"
    qwen_dg_file = DIFF_DIR / "qwen_vl_ios1k_dg_train.json"

    # Build SOM records
    som_records = []
    dg_records = []
    qwen_som_items = []
    qwen_dg_items = []

    for el in elements:
        tid = el["target_id"]
        cx = el["center"]["x"]
        cy = el["center"]["y"]
        inst = el["instruction"]
        typ = el["type"].lower()

        # SOM record
        som_records.append({
            "language": "English",
            "step_name": f"ios1k_{stem}_{typ}",
            "instruction": inst,
            "action": "click",
            "target_id": tid,
            "center_x": cx,
            "center_y": cy,
            "screenshot_path": str(img_path),
            "annotated_image_path": str(som_img_path),
            "decision": {
                "status": "ground_truth_ios1k_som",
                "model": "ground_truth",
                "target_id": tid,
                "confidence": 1.0,
                "rationale": f"Box [{tid}] contains the target for '{inst}'.",
                "raw_output": json.dumps({"target_id": tid, "confidence": 1.0})
            },
            "tap_x": cx,
            "tap_y": cy,
            "timestamp": f"ios1k_som_{stem}_{tid}_{cx}_{cy}",
            "is_correct": True,
            "is_action_successful": True,
            "gt_x": cx,
            "gt_y": cy,
            "discarded": False,
            "dataset_source": "iOS-1K-Mobile-UI-Dataset",
            "ground_truth_id": tid
        })

        # DG record
        norm_x = int(cx * 1000 / 828)
        norm_y = int(cy * 1000 / 1792)
        dg_records.append({
            "language": "English",
            "step_name": f"ios1k_{stem}_{typ}",
            "instruction": inst,
            "action": "click",
            "target_id": tid,
            "center_x": cx,
            "center_y": cy,
            "screenshot_path": str(img_path),
            "annotated_image_path": str(img_path),
            "decision": {
                "status": "ground_truth_ios1k_dg",
                "model": "ground_truth",
                "point": [norm_y, norm_x],
                "confidence": 1.0,
                "rationale": f"Target for '{inst}'.",
                "raw_output": json.dumps({"point": [norm_y, norm_x], "confidence": 1.0})
            },
            "tap_x": cx,
            "tap_y": cy,
            "timestamp": f"ios1k_dg_{stem}_{tid}_{cx}_{cy}",
            "is_correct": True,
            "is_action_successful": True,
            "gt_x": cx,
            "gt_y": cy,
            "discarded": False,
            "dataset_source": "iOS-1K-Mobile-UI-Dataset",
            "ground_truth_id": tid
        })

        # Qwen-VL SOM
        qwen_som_items.append({
            "id": f"ios1k_som_{stem}_{tid}",
            "image": str(som_img_path),
            "conversations": [
                {"from": "user", "value": f"<image>\n{inst}"},
                {"from": "gpt", "value": str(tid)}
            ]
        })

        # Qwen-VL DG
        qwen_dg_items.append({
            "id": f"ios1k_dg_{stem}_{tid}",
            "image": str(img_path),
            "conversations": [
                {"from": "user", "value": f"<image>\n{inst}"},
                {"from": "gpt", "value": json.dumps({"point": [norm_y, norm_x]})}
            ]
        })

    # Helper to insert or replace in JSONL
    def _update_jsonl(filepath, new_recs):
        if not filepath.exists():
            return
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        filtered = []
        insert_idx = None
        for idx, l in enumerate(lines):
            if f"_{stem}_" in l or f"ios1k_{stem}" in l:
                if insert_idx is None:
                    insert_idx = len(filtered)
                continue
            filtered.append(l)

        if insert_idx is None:
            prev_num = int(stem) - 1
            for p in range(prev_num, 0, -1):
                p_str = f"{p:04d}"
                for idx, l in enumerate(filtered):
                    if f"_{p_str}_" in l or f"ios1k_{p_str}" in l:
                        insert_idx = idx + 1
                if insert_idx is not None:
                    break

        if insert_idx is None:
            insert_idx = len(filtered)

        for r in reversed(new_recs):
            filtered.insert(insert_idx, json.dumps(r, ensure_ascii=False) + "\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(filtered)

    # Helper to insert or replace in Qwen JSON
    def _update_qwen(filepath, new_items, is_som=True):
        if not filepath.exists():
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        identifier = f"{stem}_som.png" if is_som else f"{stem}.png"
        filtered = []
        insert_idx = None
        for idx, it in enumerate(data):
            img_str = it.get("image", "") or (it.get("images", [""])[0] if it.get("images") else "")
            if identifier in img_str:
                if insert_idx is None:
                    insert_idx = len(filtered)
                continue
            filtered.append(it)

        if insert_idx is None:
            prev_num = int(stem) - 1
            for p in range(prev_num, 0, -1):
                p_str = f"{p:04d}_som.png" if is_som else f"{p:04d}.png"
                for idx, it in enumerate(filtered):
                    img_s = it.get("image", "") or (it.get("images", [""])[0] if it.get("images") else "")
                    if p_str in img_s:
                        insert_idx = idx + 1
                if insert_idx is not None:
                    break

        if insert_idx is None:
            insert_idx = len(filtered)

        for item in reversed(new_items):
            filtered.insert(insert_idx, item)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

    _update_jsonl(som_file, som_records)
    _update_jsonl(dg_file, dg_records)
    _update_qwen(qwen_som_file, qwen_som_items, is_som=True)
    _update_qwen(qwen_dg_file, qwen_dg_items, is_som=False)


# ==============================================================================
# 7. Main CLI Execution Pipeline
# ==============================================================================
def process_ios_dataset(dataset_dir, output_dir, screens=None, mode="cv_ai", ai_model="gpt-4o-mini", force=False):
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    som_dir = output_dir / "som_images"
    som_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images" if (dataset_dir / "images").exists() else dataset_dir
    annotations_dir = dataset_dir / "annotations" if (dataset_dir / "annotations").exists() else dataset_dir

    reviewed_stems = get_reviewed_stems()
    print(f"🔒 Identified {len(reviewed_stems)} screens protected by human review in finetune_dataset.jsonl")

    # Determine target screens
    all_imgs = sorted([f for f in images_dir.glob("*.png") if f.stat().st_size > 5000])

    if screens:
        target_stems = set()
        for part in screens.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                for num in range(int(start), int(end) + 1):
                    target_stems.add(f"{num:04d}")
            else:
                target_stems.add(f"{int(part):04d}")
        image_files = [f for f in all_imgs if f.stem in target_stems]
    else:
        image_files = all_imgs

    print(f"🎯 Processing {len(image_files)} screens in mode '{mode}' (AI model: {ai_model})...")

    success_count = 0
    for img_p in image_files:
        stem = img_p.stem
        ann_p = annotations_dir / f"{stem}.json"
        som_p = som_dir / f"{stem}_som.png"

        # Check if protected by human review
        if stem in reviewed_stems and not force:
            print(f"  ⏩ [Protected] Screen {stem} is already reviewed in finetune_dataset.jsonl. Skipping!")
            continue

        print(f"  [CV+AI] Annotating Screen {stem}...")
        elements = annotate_screen_cv_ai(
            img_p,
            json_output_path=ann_p,
            use_ai=(mode == "cv_ai"),
            model=ai_model
        )

        if not elements:
            print(f"  ⚠️ Warning: No elements found for screen {stem}")
            continue

        draw_som_image(img_p, elements, som_p)
        update_datasets_for_screen(stem, elements, img_p, som_p)
        print(f"  ✅ Screen {stem}: {len(elements)} elements annotated & synced.")
        success_count += 1
        import time
        time.sleep(1.2)

    print(f"\n🎉 Finished processing {success_count} screens successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure CV + AI UI Grounding Pipeline for iOS-1K")
    parser.add_argument("--dataset-dir", default=str(DATASET_ROOT), help="Root directory of iOS-1K dataset")
    parser.add_argument("--output-dir", default=str(DIFF_DIR), help="Output directory for diff and SoM images")
    parser.add_argument("--screens", default=None, help="Target screen or range, e.g. '0301', '0301-0310'")
    parser.add_argument("--mode", choices=["cv_ai", "rule"], default="cv_ai", help="Annotation mode (cv_ai or rule)")
    parser.add_argument("--ai-model", default="gpt-4o-mini", help="Vision AI model for semantic reasoning")
    parser.add_argument("--force", action="store_true", help="Force re-annotation of protected screens")

    args = parser.parse_args()
    process_ios_dataset(
        args.dataset_dir,
        args.output_dir,
        screens=args.screens,
        mode=args.mode,
        ai_model=args.ai_model,
        force=args.force
    )
