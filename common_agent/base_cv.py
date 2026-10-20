"""
Core Computer Vision & Geometric Utilities for Visual Agents.
Unified implementation for both Set-of-Mark (SoM) and Direct Grounding (DG) modes.
"""

import os
import re
import json
import base64
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)
    if x_right <= x_left or y_bottom <= y_top:
        return 0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0


def _is_concentric_duplicate(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)
    if x_right <= x_left or y_bottom <= y_top:
        return False

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = aw * ah
    area_b = bw * bh
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    if larger / smaller > 2.5:
        return False
    return (intersection / smaller) > 0.70


def _filter_enclosing_containers(boxes, min_container_area=80000):
    filtered = []
    for b in boxes:
        bx, by, bw, bh = b
        area = bw * bh
        if area > min_container_area:
            inside_count = sum(
                1 for o in boxes
                if o != b and o[0] >= bx - 5 and o[1] >= by - 5
                and o[0] + o[2] <= bx + bw + 5 and o[1] + o[3] <= by + bh + 5
            )
            if inside_count >= 1:
                continue
        filtered.append(b)
    return filtered


def _merge_horizontal_boxes(boxes, max_gap=40):
    changed = True
    curr = list(boxes)
    while changed:
        changed = False
        nb = []
        used = [False] * len(curr)
        for i in range(len(curr)):
            if used[i]:
                continue
            ax, ay, aw, ah = curr[i]
            for j in range(i + 1, len(curr)):
                if used[j]:
                    continue
                bx, by, bw, bh = curr[j]
                y_top = max(ay, by)
                y_bottom = min(ay + ah, by + bh)
                v_overlap = max(0, y_bottom - y_top)
                min_h = min(ah, bh)
                if min_h > 0 and (v_overlap / min_h) >= 0.55:
                    gap = max(ax, bx) - min(ax + aw, bx + bw)
                    if 0 <= gap <= max_gap:
                        nx = min(ax, bx)
                        ny = min(ay, by)
                        nw = max(ax + aw, bx + bw) - nx
                        nh = max(ay + ah, by + bh) - ny
                        ax, ay, aw, ah = nx, ny, nw, nh
                        used[j] = True
                        changed = True
            nb.append((ax, ay, aw, ah))
            used[i] = True
        curr = nb
    return curr


def _merge_adjacent_icon_bars(boxes, max_v_gap=25):
    """
    Merges stacked icon bars (e.g. 3 lines of a hamburger menu icon)
    into a single unified icon bounding box.
    """
    changed = True
    curr = list(boxes)
    while changed:
        changed = False
        nb = []
        used = [False] * len(curr)
        for i in range(len(curr)):
            if used[i]:
                continue
            ax, ay, aw, ah = curr[i]
            for j in range(i + 1, len(curr)):
                if used[j]:
                    continue
                bx, by, bw, bh = curr[j]
                if abs(ax - bx) <= 15 and abs(aw - bw) <= 20:
                    v_gap = max(ay, by) - min(ay + ah, by + bh)
                    if 0 <= v_gap <= max_v_gap:
                        ax, ay, aw, ah = (
                            min(ax, bx),
                            min(ay, by),
                            max(ax + aw, bx + bw) - min(ax, bx),
                            max(ay + ah, by + bh) - min(ay, by),
                        )
                        used[j] = True
                        changed = True
            nb.append((ax, ay, aw, ah))
            used[i] = True
        curr = nb
    return curr


def _merge_vertical_text_lines(boxes, max_v_gap=16):
    """
    Merges vertically adjacent text lines within a paragraph/block of text
    (e.g. multiline descriptions or hints) so that wrapped line fragments
    are not mistakenly detected as isolated interactive buttons.
    """
    changed = True
    curr = list(boxes)
    while changed:
        changed = False
        nb = []
        used = [False] * len(curr)
        for i in range(len(curr)):
            if used[i]:
                continue
            ax, ay, aw, ah = curr[i]
            for j in range(i + 1, len(curr)):
                if used[j]:
                    continue
                bx, by, bw, bh = curr[j]
                if ay <= by:
                    top_box, bot_box = (ax, ay, aw, ah), (bx, by, bw, bh)
                else:
                    top_box, bot_box = (bx, by, bw, bh), (ax, ay, aw, ah)

                tax, tay, taw, tah = top_box
                bax, bay, baw, bah = bot_box

                v_gap = bay - (tay + tah)
                if 0 <= v_gap <= max_v_gap and tah <= 60 and bah <= 60:
                    x_left = max(tax, bax)
                    x_right = min(tax + taw, baw + bax)
                    overlap_w = max(0, x_right - x_left)
                    min_w = min(taw, baw)
                    if min_w > 0 and (overlap_w / min_w) >= 0.8:
                        nx = min(ax, bx)
                        ny = min(ay, by)
                        nw = max(ax + aw, bx + bw) - nx
                        nh = max(ay + ah, by + bh) - ny
                        ax, ay, aw, ah = nx, ny, nw, nh
                        used[j] = True
                        changed = True
            nb.append((ax, ay, aw, ah))
            used[i] = True
        curr = nb
    return curr


def _detect_modal_dialog(image):
    """
    Universally detects whether a modal dialog, popup, or alert card is on screen.
    Returns (x, y, w, h) of the dialog or None if no modal dialog is active.
    A modal dialog is a floating card with distinct margins on both sides of the screen.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        # Modal dialog MUST be floating (distinct margins on left and right; not full width)
        if x < 40 or x + bw > w - 40:
            continue
        if y < 100 or y + bh > h - 100:
            continue
        # Area between 15% and 70% of total screen
        if 0.15 * w * h < area < 0.70 * w * h:
            aspect = bw / bh
            if 0.5 <= aspect <= 2.0:
                cx = x + bw / 2
                cy = y + bh / 2
                if abs(cx - w / 2) < 0.15 * w and 0.25 * h < cy < 0.75 * h:
                    # Check whether this is a solid color dialog card (e.g. >70% uniform background)
                    roi = image[y:y+bh, x:x+bw]
                    median_color = np.median(roi.reshape(-1, 3), axis=0)
                    color_diff = np.linalg.norm(roi.astype(float) - median_color, axis=2)
                    solid_ratio = (color_diff < 12).mean()
                    if solid_ratio >= 0.70:
                        candidates.append((x, y, bw, bh))

    if not candidates:
        return None
    return max(candidates, key=lambda b: b[2] * b[3])


def _intersection_over_area(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)
    if x_right <= x_left or y_bottom <= y_top:
        return 0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    smaller_area = min(aw * ah, bw * bh)
    return intersection / smaller_area if smaller_area else 0


def _deduplicate_boxes(boxes, overlap_threshold=0.82):
    sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
    kept = []
    for box in sorted_boxes:
        if all(_intersection_over_area(box, kept_box) < overlap_threshold for kept_box in kept):
            kept.append(box)

    return sorted(kept, key=lambda box: (box[1], box[0]))


def annotate_ui_elements_som(
    image_path,
    output_path=None,
    min_box_size=18,
    min_area=350,
    max_area_ratio=0.35,
    canny_low=50,
    canny_high=150,
    **kwargs
):
    """
    CV Detection for Set-of-Mark (SoM) mode:
    Identifies buttons, fields, and icons, applies intelligent merging & modal isolation,
    draws red numbered bounding boxes [1], [2], and returns (elements_map, annotated_img_path, elements).
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, canny_low, canny_high)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)

    threshold_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    modal = _detect_modal_dialog(image)
    boxes = []
    if modal:
        # Modal dialog active: ONLY mark elements within the dialog, ignore background
        mx, my, mw, mh = modal
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7))
        t_mask = cv2.morphologyEx(threshold_mask.copy(), cv2.MORPH_CLOSE, close_kernel, iterations=1)
        combined = cv2.bitwise_or(edge_mask, t_mask)
        contours, _ = cv2.findContours(combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Only allow bounding boxes within dialog boundaries
            if x >= mx - 10 and y >= my - 10 and x + w <= mx + mw + 10 and y + h <= my + mh + 10:
                area = w * h
                if area > mw * mh * 0.8:
                    continue  # Ignore dialog background card itself
                if w < 22 or h < 18 or area < 350:
                    continue
                raw_boxes.append((x, y, w, h))

        if raw_boxes:
            boxes = _merge_horizontal_boxes(raw_boxes, max_gap=40)
            sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3])
            kept = []
            for b in sorted_boxes:
                if not any(_is_concentric_duplicate(b, k) or _iou(b, k) > 0.6 for k in kept):
                    kept.append(b)
            boxes = sorted(kept, key=lambda box: (box[1], box[0]))

    # Fallback to standard view if no modal dialog was detected or fewer than 2 elements were found
    if not boxes or len(boxes) < 2:
        # Standard view without dialog: clean pipeline for standard screens
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        t_mask = cv2.morphologyEx(threshold_mask.copy(), cv2.MORPH_CLOSE, close_kernel, iterations=1)
        combined = cv2.bitwise_or(edge_mask, t_mask)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = width * height * max_area_ratio
        raw_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h else 0
            # Ignore iOS status bar (time, Wi-Fi, battery) at the top
            if y < 110:
                continue
            if w < 14 or h < 8 or area < 250 or area > max_area:
                continue
            if aspect_ratio > 14 or aspect_ratio < 0.06:
                continue
            raw_boxes.append((x, y, w, h))

        # Merge adjacent icon lines, horizontal words, and multiline text
        boxes = _merge_adjacent_icon_bars(raw_boxes, max_v_gap=25)
        boxes = _merge_horizontal_boxes(boxes, max_gap=45)
        boxes = _merge_vertical_text_lines(boxes, max_v_gap=16)
        boxes = [b for b in boxes if b[2] >= 18 and b[3] >= 16 and (b[2] * b[3]) >= 400]

        sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        kept = []
        for box in sorted_boxes:
            if all(_intersection_over_area(box, kept_box) < 0.82 for kept_box in kept):
                kept.append(box)
        boxes = sorted(kept, key=lambda box: (box[1], box[0]))

    marked = image.copy()
    elements = []
    elements_map = {}

    for index, (x, y, w, h) in enumerate(boxes, start=1):
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        label = f"[{index}]"
        elements.append({
            "id": index,
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "center": {"x": center_x, "y": center_y},
            "area": int(w * h),
        })
        elements_map[str(index)] = (center_x, center_y)

        # Draw red bounding rectangle and label with white background
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label_origin = (x, max(18, y - 6))
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            marked,
            (label_origin[0], label_origin[1] - label_h - 5),
            (label_origin[0] + label_w + 6, label_origin[1] + 3),
            (255, 255, 255),
            -1,
        )
        cv2.putText(marked, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # Save annotated image
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_som.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), marked)

    # Save metadata JSON alongside annotated image
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)

    return elements_map, str(output_path), elements


annotate_ui_elements = annotate_ui_elements_som


def annotate_ui_elements_dg(image_path, output_path=None, **kwargs):
    """
    Direct Grounding (DG) mode:
    No OpenCV boxes and no Set-of-Mark labels.
    The original clean screenshot is passed directly to the vision model.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_target_path = image_path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path != image_path:
            cv2.imwrite(str(output_path), image)
        img_target_path = output_path

    elements = []
    dynamic_elements_map = {}
    return dynamic_elements_map, str(img_target_path), elements


def encode_image(image_path, max_dim=None):
    """
    Encodes an image to a base64 string, mime type, and data URL.
    Proportionally resizes the image so that its longest edge does not exceed max_dim (default 1600px).
    Configurable via MAX_IMAGE_EDGE environment variable (set <= 0 to disable resizing).
    """
    if max_dim is None:
        try:
            max_dim = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
        except ValueError:
            max_dim = 1600

    image_file = Path(image_path)
    suffix = image_file.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"

    # If resizing is disabled, read raw file directly
    if max_dim <= 0:
        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
        return encoded, mime_type, f"data:{mime_type};base64,{encoded}"

    img = cv2.imread(str(image_path))
    if img is None:
        # Fallback to reading raw file if cv2 fails to read
        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
        return encoded, mime_type, f"data:{mime_type};base64,{encoded}"

    h, w = img.shape[:2]
    longest = max(h, w)

    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        print(f"🖼️ [Image Upload] Resized for VLM: {w}x{h} -> {new_w}x{new_h} (max_dim={max_dim})")
        # cv2.INTER_AREA is optimal for image downsampling, preserving sharp text and markings
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        print(f"🖼️ [Image Upload] Image within limit: {w}x{h} (max_dim={max_dim})")

    ext = ".jpg" if mime_type == "image/jpeg" else ".png"
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 90]
    success, buffer = cv2.imencode(ext, img, encode_params)
    if not success:
        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    else:
        encoded = base64.b64encode(buffer).decode("ascii")

    data_url = f"data:{mime_type};base64,{encoded}"
    return encoded, mime_type, data_url


def is_page_changed(img_path_a, img_path_b, threshold=0.005):
    """
    Compares pre-tap and post-tap screenshots using absolute grayscale diff
    to determine if screen layout changed.
    """
    img_a = cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return True

    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = cv2.absdiff(img_a, img_b)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    non_zero = cv2.countNonZero(thresh)
    total_pixels = img_a.shape[0] * img_a.shape[1]
    ratio = non_zero / total_pixels
    return ratio > threshold


def parse_dg_coordinates(output_text, image_width=1206, image_height=2622, is_normalized_1000=False):
    """
    Parses direct grounding interaction coordinates (target_x, target_y) from model text.
    Handles:
    1. JSON schemas: {"point": [x, y]}, {"coordinates": [x, y]}, {"x": x, "y": y}, {"bbox": ...}
    2. Qwen-VL Grounding tags: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
    3. Regex matching: [x, y], (x, y)
    4. Auto-scales normalized [0, 1] or [0, 1000] (if is_normalized_1000=True) to absolute pixels.
    """
    target_x = None
    target_y = None
    confidence = 1.0
    rationale = output_text

    # 1. JSON parsing
    parsed_json = None
    try:
        parsed_json = json.loads(output_text)
    except Exception:
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed_json = json.loads(output_text[start:end + 1])
            except Exception:
                pass

    if isinstance(parsed_json, dict):
        confidence = float(parsed_json.get("confidence", 1.0))
        rationale = parsed_json.get("rationale") or output_text

        if "point" in parsed_json and isinstance(parsed_json["point"], (list, tuple)) and len(parsed_json["point"]) >= 2:
            target_x = float(parsed_json["point"][0])
            target_y = float(parsed_json["point"][1])
        elif "coordinates" in parsed_json and isinstance(parsed_json["coordinates"], (list, tuple)) and len(parsed_json["coordinates"]) >= 2:
            target_x = float(parsed_json["coordinates"][0])
            target_y = float(parsed_json["coordinates"][1])
        elif "target_point" in parsed_json and isinstance(parsed_json["target_point"], (list, tuple)) and len(parsed_json["target_point"]) >= 2:
            target_x = float(parsed_json["target_point"][0])
            target_y = float(parsed_json["target_point"][1])
        elif "x" in parsed_json and "y" in parsed_json:
            target_x = float(parsed_json["x"])
            target_y = float(parsed_json["y"])
        elif "center" in parsed_json:
            center = parsed_json["center"]
            if isinstance(center, dict) and "x" in center and "y" in center:
                target_x = float(center["x"])
                target_y = float(center["y"])
            elif isinstance(center, (list, tuple)) and len(center) >= 2:
                target_x = float(center[0])
                target_y = float(center[1])
        elif "bbox" in parsed_json and isinstance(parsed_json["bbox"], (list, tuple)) and len(parsed_json["bbox"]) >= 4:
            b = parsed_json["bbox"]
            target_x = (float(b[1]) + float(b[3])) / 2.0
            target_y = (float(b[0]) + float(b[2])) / 2.0
        elif "bbox_2d" in parsed_json and isinstance(parsed_json["bbox_2d"], (list, tuple)) and len(parsed_json["bbox_2d"]) >= 4:
            b = parsed_json["bbox_2d"]
            target_x = (float(b[0]) + float(b[2])) / 2.0
            target_y = (float(b[1]) + float(b[3])) / 2.0

    # 2. Qwen-VL Standard Grounding Tag: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
    if target_x is None or target_y is None:
        box_pattern = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', output_text)
        if box_pattern:
            y1, x1, y2, x2 = map(int, box_pattern.groups())
            norm_cx = (x1 + x2) / 2.0 / 1000.0
            norm_cy = (y1 + y2) / 2.0 / 1000.0
            target_x = norm_cx * image_width
            target_y = norm_cy * image_height

    # 3. Regex Fallback [x, y] or (x, y)
    if target_x is None or target_y is None:
        point_match = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', output_text)
        if point_match:
            target_x = float(point_match.group(1))
            target_y = float(point_match.group(2))
        else:
            coord_match = re.search(r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)', output_text)
            if coord_match:
                target_x = float(coord_match.group(1))
                target_y = float(coord_match.group(2))
            else:
                relaxed_match = re.search(r'["\']?point["\']?\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', output_text)
                if relaxed_match:
                    target_x = float(relaxed_match.group(1))
                    target_y = float(relaxed_match.group(2))

    # 4. Scaling
    if target_x is not None and target_y is not None:
        if 0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0:
            target_x = target_x * image_width
            target_y = target_y * image_height
        elif is_normalized_1000 and target_x <= 1000 and target_y <= 1000 and (image_height > 1200 or image_width > 1000):
            target_x = (target_x / 1000.0) * image_width
            target_y = (target_y / 1000.0) * image_height

        target_x = int(max(0, min(image_width, round(target_x))))
        target_y = int(max(0, min(image_height, round(target_y))))
        return target_x, target_y, confidence, rationale

    return None, None, 0.0, rationale


def heuristic_fallback_som(instruction, elements):
    """
    Universal heuristic fallback for mobile Set-of-Mark elements:
    Selects the most suitable UI element based on mobile UI/UX positioning conventions
    (header, bottom navigation tabs, and form input standards) if model inference fails.
    """
    if not elements:
        return None

    physical_h = max(el["bbox"]["y"] + el["bbox"]["height"] for el in elements)
    physical_w = max(el["bbox"]["x"] + el["bbox"]["width"] for el in elements)
    if physical_w < 800:
        physical_w = 1206
    if physical_h < 1500:
        physical_h = 2622

    def find_closest(tx, ty):
        def distance(el):
            return (el["center"]["x"] - tx)**2 + (el["center"]["y"] - ty)**2
        return min(elements, key=distance)["id"]

    instruction_lower = instruction.lower()

    # 1. Top Navigation / Header (Back, Drawer, Menu on top left)
    if any(keyword in instruction_lower for keyword in ["back", "back icon", "back arrow", "return", "go back", "drawer", "menu", "hamburger", "avatar", "top left corner"]):
        return find_closest(physical_w * 0.08, physical_h * 0.096)

    # 2. Top Actions / Close / Cancel on top right
    if any(keyword in instruction_lower for keyword in ["close", "close button", "dismiss", "x", "cancel", "exit", "done", "top right corner"]):
        top_right_elems = [e for e in elements if e["center"]["x"] > physical_w * 0.6 and e["center"]["y"] < physical_h * 0.3]
        if top_right_elems:
            return min(top_right_elems, key=lambda e: (e["center"]["x"] - physical_w * 0.92)**2 + (e["center"]["y"] - physical_h * 0.096)**2)["id"]
        return find_closest(physical_w * 0.92, physical_h * 0.096)

    # 3. Bottom Navigation bar (standard 3 to 5 tabs at the bottom)
    bottom_y = physical_h - 150
    is_tab_query = any(k in instruction_lower for k in ["tab", "bottom bar", "tab bar", "nav bar", "navigation"])

    # 3.1 Tab 1 / Far left (~12-15% x): Home / Start / Main / Dashboard
    if any(k in instruction_lower for k in ["home", "start", "main", "dashboard"]) or (is_tab_query and "1" in instruction_lower):
        return find_closest(physical_w * 0.15, bottom_y)

    # 3.2 Tab 2 / Middle-left (~32-35% x): Search / Explore / Discover / Scan
    if any(k in instruction_lower for k in ["search", "explore", "discover", "find", "scan"]) or (is_tab_query and "2" in instruction_lower):
        return find_closest(physical_w * 0.35, bottom_y)

    # 3.3 Tab 3 / Center (~50% x): Central action / Feed / Chat / Message / Overview / Triage
    if any(k in instruction_lower for k in ["chat", "message", "feed", "create", "center", "overview", "triagen", "triages", "inbox"]) or (is_tab_query and "3" in instruction_lower):
        return find_closest(physical_w * 0.50, bottom_y)

    # 3.4 Tab 4 / Middle-right (~68-70% x): Notifications / Activity / Bell / Alerts
    if any(k in instruction_lower for k in ["notification", "bell", "alerts", "activity", "status"]) or (is_tab_query and "4" in instruction_lower):
        return find_closest(physical_w * 0.70, bottom_y)

    # 3.5 Tab 5 / Far right (~88% x): Profile / User / Settings / Account / Vehicle
    if any(k in instruction_lower for k in ["profile", "account", "settings", "user", "me", "vehicle", "@"]) or (is_tab_query and "5" in instruction_lower):
        return find_closest(physical_w * 0.88, bottom_y)

    # 4. Form and interactive input elements
    # 4.1 Input text fields (upper / middle third)
    if any(k in instruction_lower for k in ["input", "text field", "field", "type", "search bar", "edit text", "token"]):
        upper_inputs = [e for e in elements if e["center"]["y"] < physical_h * 0.5]
        if upper_inputs:
            return min(upper_inputs, key=lambda e: (e["center"]["x"] - physical_w * 0.5)**2 + (e["center"]["y"] - physical_h * 0.25)**2)["id"]
        return find_closest(physical_w * 0.5, physical_h * 0.25)

    # 4.2 Dropdown / Picker / Selection list (middle area)
    if any(k in instruction_lower for k in ["dropdown", "select", "picker", "choice", "option", "menu", "language"]):
        return find_closest(physical_w * 0.5, physical_h * 0.45)

    # 4.3 Confirmation, Submit, or Save buttons (lower area or primary action button)
    if any(k in instruction_lower for k in ["submit", "continue", "save", "ok", "confirm", "login", "sign in", "register", "next", "send", "apply"]):
        action_buttons = [e for e in elements if e["center"]["y"] > physical_h * 0.4]
        if action_buttons:
            return min(action_buttons, key=lambda e: abs(e["center"]["x"] - physical_w * 0.5) + abs(e["center"]["y"] - physical_h * 0.85))["id"]
        return find_closest(physical_w * 0.5, physical_h * 0.85)

    # 5. Default fallback: element closest to screen center
    return find_closest(physical_w * 0.5, physical_h * 0.5)


def heuristic_fallback_dg(instruction, image_size=(1206, 2622)):
    """Fallback coordinates for Direct Grounding based on iOS mobile UI conventions."""
    physical_w, physical_h = image_size
    instruction_lower = instruction.lower()

    # 1. Top Navigation / Back
    if any(k in instruction_lower for k in ["back", "back icon", "back arrow", "return", "go back", "top left corner"]):
        return int(physical_w * 0.08), int(physical_h * 0.096)

    # 2. Top Actions / Menu / Drawer / Close
    if any(k in instruction_lower for k in ["menu", "drawer", "hamburger", "avatar"]):
        return int(physical_w * 0.08), int(physical_h * 0.096)

    if any(k in instruction_lower for k in ["close", "dismiss", "x", "cancel", "top right corner"]):
        return int(physical_w * 0.92), int(physical_h * 0.096)

    # 3. Bottom Navigation bar (tabs)
    bottom_y = int(physical_h - 75)
    if any(k in instruction_lower for k in ["home", "start", "main", "dashboard"]):
        return int(physical_w * 0.15), bottom_y
    if any(k in instruction_lower for k in ["search", "explore", "discover", "find", "scan"]):
        return int(physical_w * 0.35), bottom_y
    if any(k in instruction_lower for k in ["chat", "message", "feed", "inbox"]):
        return int(physical_w * 0.50), bottom_y
    if any(k in instruction_lower for k in ["notification", "bell", "alerts", "activity"]):
        return int(physical_w * 0.70), bottom_y
    if any(k in instruction_lower for k in ["profile", "account", "settings", "user"]):
        return int(physical_w * 0.88), bottom_y

    # 4. Confirmation / Primary action buttons
    if any(k in instruction_lower for k in ["submit", "continue", "save", "ok", "confirm", "login", "sign in"]):
        return int(physical_w * 0.5), int(physical_h * 0.85)

    return int(physical_w * 0.5), int(physical_h * 0.5)


def write_interaction_log(
    log_dir,
    mode,
    language,
    step_name,
    instruction,
    action,
    target_id,
    center_x,
    center_y,
    screenshot_path,
    annotated_image_path,
    decision,
    tap_x=None,
    tap_y=None,
):
    """
    Standardized logger:
    - In SoM mode: writes to log_dir / 'interaction_log.jsonl'
    - In DG mode:  writes to log_dir / 'interaction_log_DG.jsonl'
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    filename = "interaction_log_DG.jsonl" if mode == "dg" else "interaction_log.jsonl"
    log_file = log_path / filename

    record = {
        "language": language,
        "step_name": step_name,
        "instruction": instruction,
        "action": action,
        "mode": "direct_grounding" if mode == "dg" else "set_of_mark",
        "target_id": target_id if target_id is not None else 1,
        "center_x": center_x,
        "center_y": center_y,
        "screenshot_path": str(screenshot_path),
        "annotated_image_path": str(annotated_image_path),
        "decision": decision,
        "tap_x": tap_x,
        "tap_y": tap_y,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)
