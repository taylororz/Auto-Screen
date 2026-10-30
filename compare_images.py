import argparse
import base64
import json
import html
import os
import ssl
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from urllib import error, request

import cv2
import numpy as np
import re
import time


DEFAULT_AGENT_MODEL = "gpt-4o-mini"


def _merge_close_boxes(boxes, padding_x=22, padding_y=10):
    """Iteratively merges bounding boxes that are horizontally adjacent or vertically close in the same UI component."""
    if not boxes:
        return []
    merged = list(boxes)
    while True:
        changed = False
        new_merged = []
        for box in merged:
            bx, by, bw, bh = box
            matched_idx = -1
            for i, m in enumerate(new_merged):
                mx, my, mw, mh = m
                if not (bx > mx + mw + padding_x or mx > bx + bw + padding_x or
                        by > my + mh + padding_y or my > by + bh + padding_y):
                    matched_idx = i
                    break
            if matched_idx >= 0:
                mx, my, mw, mh = new_merged[matched_idx]
                nx = min(mx, bx)
                ny = min(my, by)
                nw = max(mx + mw, bx + bw) - nx
                nh = max(my + mh, by + bh) - ny
                new_merged[matched_idx] = (nx, ny, nw, nh)
                changed = True
            else:
                new_merged.append(box)
        merged = new_merged
        if not changed:
            break
    return sorted(merged, key=lambda b: (b[1], b[0]))


def _draw_diff_highlight(image, boxes, color=(0, 0, 255)):
    """Draws semi-transparent overlay highlights and crisp border rectangles for diff boxes. If no boxes, returns clean image."""
    if not boxes:
        return image.copy()
    result = image.copy()
    overlay = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    cv2.addWeighted(overlay, 0.22, result, 0.78, 0, result)
    return result


def _apply_ignore_regions(mask, ignore_regions):
    """Zeroes the configured ignore regions ([x, y, w, h] in screenshot pixels) in the comparison mask."""
    for (rx, ry, rw, rh) in ignore_regions or []:
        mask[max(0, ry):max(0, ry + rh), max(0, rx):max(0, rx + rw)] = 0
    return mask


def _draw_ignored_regions(image, ignore_regions):
    """Outlines ignored regions in gray so the report shows which areas were not compared."""
    for (rx, ry, rw, rh) in ignore_regions or []:
        cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (150, 150, 150), 2)
        cv2.putText(image, "ignored", (rx + 6, ry + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    return image


def _ignore_rects_for_page(ignore_regions, folder_name, page_name):
    """Returns the rectangles of the configured ignore regions that apply to a folder and page.
    A region without "pages" or "folders" applies to all pages or folders."""
    rects = []
    for region in ignore_regions or []:
        pages = region.get("pages")
        folders = region.get("folders")
        if pages and page_name not in pages:
            continue
        if folders and folder_name not in folders:
            continue
        rect = region.get("rect")
        if rect and len(rect) == 4:
            rects.append(tuple(int(v) for v in rect))
    return rects


def compare_pixel_matching(baseline_path, actual_path, diff_output_path, min_box_size=14, color_threshold=38, ignore_top=140, ignore_bottom=70, ignore_regions=None, boxes_out=None):
    """
    Color-tolerant pixel matching: calculates channel-wise color difference to eliminate anti-aliasing noise,
    ignores system status bar, connects line-level differences, and merges bounding boxes.
    Only marks genuinely different regions; completely ignores identical elements.
    Returns: (diff_count, execution_time_ms, output_path_str)
    """
    start_time = time.time()
    
    base_img = cv2.imread(str(baseline_path))
    actual_img = cv2.imread(str(actual_path))

    if base_img is None or actual_img is None:
        print("Error: One or both images could not be read. Please verify file paths.")
        return 0, 0, None

    if base_img.shape != actual_img.shape:
        actual_img = cv2.resize(actual_img, (base_img.shape[1], base_img.shape[0]))

    h, w = base_img.shape[:2]

    # 1. Color-aware difference: maximum channel difference suppresses sub-pixel anti-aliasing
    diff_channels = cv2.absdiff(base_img, actual_img)
    max_diff = np.max(diff_channels, axis=2)

    # 2. Status bar & home indicator mask
    mask = np.ones((h, w), dtype=np.uint8)
    if ignore_top > 0 and h > ignore_top:
        mask[:ignore_top, :] = 0
    if ignore_bottom > 0 and h > ignore_bottom:
        mask[h - ignore_bottom:, :] = 0
    _apply_ignore_regions(mask, ignore_regions)

    _, diff_thresh = cv2.threshold(max_diff, color_threshold, 255, cv2.THRESH_BINARY)
    diff_thresh = cv2.bitwise_and(diff_thresh, diff_thresh, mask=mask)

    # 3. Horizontal morphological close to unify words in text rows
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, close_kernel)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw >= min_box_size and bh >= min_box_size and (bw * bh) >= 150:
            box_diff = diff_thresh[y:y+bh, x:x+bw]
            diff_pixels = np.count_nonzero(box_diff)
            # Filter noise in identical graphics: require at least 25 changed pixels and 4% density
            if diff_pixels >= 25 and (diff_pixels / (bw * bh) >= 0.04):
                raw_boxes.append((x, y, bw, bh))

    boxes = _merge_close_boxes(raw_boxes, padding_x=18, padding_y=10)
    if boxes_out is not None:
        boxes_out.extend(boxes)
    result_img = _draw_diff_highlight(actual_img, boxes, color=(0, 200, 0))
    if ignore_regions:
        result_img = _draw_ignored_regions(result_img, ignore_regions)

    pixel_output_path = Path(diff_output_path).parent / f"{Path(diff_output_path).stem}_pixel_matching.png"
    pixel_output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pixel_output_path), result_img)

    elapsed_time = (time.time() - start_time) * 1000
    diff_count = len(boxes)
    print(f"   ✅ Pixel-Matching: {diff_count} semantic differences ({elapsed_time:.2f}ms)")
    return diff_count, elapsed_time, str(pixel_output_path)


def compare_cv_grayscale(baseline_path, actual_path, diff_output_path, threshold=38, min_box_size=14, ignore_top=140, ignore_bottom=70, ignore_regions=None, boxes_out=None):
    """
    OpenCV grayscale difference method: converts to grayscale, masks system status bar,
    applies horizontal word-grouping, and clusters bounding boxes.
    Only marks genuinely different regions; completely ignores identical elements.
    Returns: (diff_count, execution_time_ms, output_path_str)
    """
    start_time = time.time()
    
    base_img = cv2.imread(str(baseline_path))
    actual_img = cv2.imread(str(actual_path))

    if base_img is None or actual_img is None:
        print("Error: One or both images could not be read. Please verify file paths.")
        return 0, 0, None

    if base_img.shape != actual_img.shape:
        actual_img = cv2.resize(actual_img, (base_img.shape[1], base_img.shape[0]))

    h, w = base_img.shape[:2]

    gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray_actual = cv2.cvtColor(actual_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_base, gray_actual)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Mask system status bar and home indicator
    mask = np.ones((h, w), dtype=np.uint8)
    if ignore_top > 0 and h > ignore_top:
        mask[:ignore_top, :] = 0
    if ignore_bottom > 0 and h > ignore_bottom:
        mask[h - ignore_bottom:, :] = 0
    _apply_ignore_regions(mask, ignore_regions)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Horizontal morphological closing to unify text lines
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw >= min_box_size and bh >= min_box_size and (bw * bh) >= 150:
            box_diff = thresh[y:y+bh, x:x+bw]
            diff_pixels = np.count_nonzero(box_diff)
            # Filter noise in identical graphics: require at least 25 changed pixels and 4% density
            if diff_pixels >= 25 and (diff_pixels / (bw * bh) >= 0.04):
                raw_boxes.append((x, y, bw, bh))

    boxes = _merge_close_boxes(raw_boxes, padding_x=18, padding_y=10)
    if boxes_out is not None:
        boxes_out.extend(boxes)
    result_img = _draw_diff_highlight(actual_img, boxes, color=(0, 0, 240))
    if ignore_regions:
        result_img = _draw_ignored_regions(result_img, ignore_regions)

    output_path = Path(diff_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_img)

    elapsed_time = (time.time() - start_time) * 1000
    diff_count = len(boxes)
    print(f"   ✅ OpenCV Grayscale: {diff_count} semantic differences ({elapsed_time:.2f}ms)")
    return diff_count, elapsed_time, str(output_path)


def compare_ui_screenshots(baseline_path, actual_path, diff_output_path, threshold=38, min_box_size=14, ignore_top=140, ignore_bottom=70, ignore_regions=None):
    print(f"🔍 Comparing: {baseline_path} vs {actual_path}")

    cv_boxes, pixel_boxes = [], []
    cv_diff_count, cv_time, cv_output = compare_cv_grayscale(
        baseline_path, actual_path, diff_output_path, threshold=threshold, min_box_size=min_box_size, ignore_top=ignore_top, ignore_bottom=ignore_bottom,
        ignore_regions=ignore_regions, boxes_out=cv_boxes
    )
    pixel_diff_count, pixel_time, pixel_output = compare_pixel_matching(
        baseline_path, actual_path, diff_output_path, color_threshold=threshold, min_box_size=min_box_size, ignore_top=ignore_top, ignore_bottom=ignore_bottom,
        ignore_regions=ignore_regions, boxes_out=pixel_boxes
    )

    print(f"   📊 Comparison: CV={cv_diff_count} Diffs, Pixel={pixel_diff_count} Diffs | CV:{cv_time:.2f}ms vs Pixel:{pixel_time:.2f}ms")

    return {
        "cv": {"count": cv_diff_count, "time_ms": cv_time, "output_path": cv_output, "boxes": cv_boxes},
        "pixel": {"count": pixel_diff_count, "time_ms": pixel_time, "output_path": pixel_output, "boxes": pixel_boxes},
    }


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


def detect_ui_elements_set_of_mark(
    image_path,
    output_path=None,
    min_box_size=18,
    min_area=350,
    max_area_ratio=0.35,
    canny_low=50,
    canny_high=150,
):
    """
    Detect likely UI controls in one screenshot and create a Set-of-Mark image.
    Returns: {"marked_image_path": str, "elements": [{id, bbox, center, area}], "count": int}
    """
    start_time = time.time()
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    max_area = width * height * max_area_ratio
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
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    combined = cv2.bitwise_or(edge_mask, threshold_mask)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h else 0
        if w < min_box_size or h < min_box_size:
            continue
        if area < min_area or area > max_area:
            continue
        if aspect_ratio > 12 or aspect_ratio < 0.08:
            continue
        boxes.append((x, y, w, h))

    boxes = _deduplicate_boxes(boxes)
    marked = image.copy()
    elements = []
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

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_som.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), marked)

    elapsed_time = (time.time() - start_time) * 1000
    print(f"   ✅ Set-of-Mark: {len(elements)} UI candidates ({elapsed_time:.2f}ms)")
    return {
        "source_image_path": str(image_path),
        "marked_image_path": str(output_path),
        "elements": elements,
        "count": len(elements),
        "time_ms": elapsed_time,
    }


def _image_data_url(image_path):
    image_file = Path(image_path)
    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    suffix = image_file.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime_type};base64,{encoded}"


def _extract_response_text(response_payload):
    if "output_text" in response_payload:
        return response_payload["output_text"]

    text_parts = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(content["text"])
    return "\n".join(text_parts)


def _parse_ai_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    return {
        "severity": "unknown",
        "summary": text.strip() or "AI returned an empty response.",
        "likely_causes": [],
        "recommended_actions": [],
        "ignore_candidate": False,
    }


def _build_ssl_context():
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()

    return ssl.create_default_context(cafile=certifi.where())


def _format_ssl_error(exc):
    return (
        f"{exc}. Python could not verify the OpenAI HTTPS certificate. "
        "Install certifi with `python3 -m pip install certifi`, or set "
        "`SSL_CERT_FILE` to a valid CA bundle path."
    )


def _post_openai_response(payload, timeout=60):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    api_request = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(api_request, timeout=timeout, context=_build_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def _openai_error_result(exc):
    if isinstance(exc, error.HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        reason = f"OpenAI API HTTP {exc.code}: {body[:500]}"
    elif isinstance(exc, error.URLError):
        reason = exc.reason
        if isinstance(reason, ssl.SSLError):
            reason = _format_ssl_error(reason)
        reason = str(reason)
    else:
        reason = str(exc)

    return {
        "status": "error",
        "reason": reason,
    }


def analyze_diff_with_ai(baseline_path, actual_path, cv_diff_path, page_name=None, cv_diff_count=0, model=None, timeout=60):
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY is not set",
            "severity": None,
            "summary": None,
            "likely_causes": [],
            "recommended_actions": [],
            "ignore_candidate": None,
        }

    if cv_diff_count == 0:
        return {
            "status": "skipped",
            "reason": "No CV differences detected",
            "severity": "pass",
            "summary": "No visual differences were detected by CV.",
            "likely_causes": [],
            "recommended_actions": [],
            "ignore_candidate": True,
        }

    model = model or os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini")
    page_label = page_name or Path(actual_path).stem
    prompt = (
        "You are reviewing visual regression screenshots for a Flutter mobile app. "
        "Compare the baseline screenshot, the actual screenshot, and the CV-annotated diff image. "
        "Decide whether the detected differences are likely harmless localization/rendering noise or a real UI regression. "
        "Return JSON only with this schema: "
        '{"severity":"pass|warn|fail","summary":"one short sentence",'
        '"likely_causes":["short cause"],"recommended_actions":["short action"],'
        '"ignore_candidate":true|false}. '
        f"Page: {page_label}. CV diff boxes: {cv_diff_count}."
    )
    payload = {
        "model": model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _image_data_url(baseline_path)},
                {"type": "input_image", "image_url": _image_data_url(actual_path)},
                {"type": "input_image", "image_url": _image_data_url(cv_diff_path)},
            ],
        }],
    }

    start_time = time.time()
    try:
        response_payload = _post_openai_response(payload, timeout=timeout)
    except Exception as exc:
        result = _openai_error_result(exc)
        result.update({
            "severity": "unknown",
            "summary": None,
            "likely_causes": [],
            "recommended_actions": [],
            "ignore_candidate": None,
        })
        return result

    parsed = _parse_ai_json(_extract_response_text(response_payload))
    parsed.setdefault("likely_causes", [])
    parsed.setdefault("recommended_actions", [])
    parsed.setdefault("ignore_candidate", False)
    parsed["status"] = "analyzed"
    parsed["model"] = model
    parsed["time_ms"] = (time.time() - start_time) * 1000
    print(f"   🤖 AI: {parsed.get('severity', 'unknown')} - {parsed.get('summary', '')}")
    return parsed


def choose_som_target_with_ai(marked_image_path, instruction, elements, language=None, model=None, timeout=60):
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY is not set",
            "target_id": None,
            "confidence": 0,
            "rationale": None,
        }

    model = model or os.environ.get("OPENAI_AGENT_MODEL", DEFAULT_AGENT_MODEL)
    element_summary = [
        {
            "id": element["id"],
            "center": element["center"],
            "bbox": element["bbox"],
        }
        for element in elements
    ]
    language_hint = f"Interface language/locale: {language}." if language else "Interface language/locale: unknown."
    prompt = (
        "You are a pure-vision mobile UI agent. The screenshot contains red Set-of-Mark labels like [1], [2], [3]. "
        "Given the user instruction, choose the single label ID that should be clicked. "
        "Use only visible UI evidence and the numbered labels. If no label matches, return target_id null. "
        "Return JSON only with this schema: "
        '{"target_id":number|null,"confidence":0.0,"rationale":"short reason","candidate_ids":[number]}. '
        f"{language_hint} User instruction: {instruction}. "
        f"Known label metadata: {json.dumps(element_summary, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _image_data_url(marked_image_path)},
            ],
        }],
    }

    start_time = time.time()
    try:
        response_payload = _post_openai_response(payload, timeout=timeout)
    except Exception as exc:
        result = _openai_error_result(exc)
        result.update({
            "target_id": None,
            "confidence": 0,
            "rationale": None,
        })
        return result

    parsed = _parse_ai_json(_extract_response_text(response_payload))
    target_id = parsed.get("target_id")
    valid_ids = {element["id"] for element in elements}
    if target_id not in valid_ids:
        target_id = None

    result = {
        "status": "analyzed",
        "model": model,
        "target_id": target_id,
        "confidence": parsed.get("confidence", 0),
        "rationale": parsed.get("rationale") or parsed.get("summary"),
        "candidate_ids": parsed.get("candidate_ids", []),
        "time_ms": (time.time() - start_time) * 1000,
    }
    print(f"   🤖 Agent: target_id={result['target_id']} confidence={result['confidence']}")
    return result


def get_element_by_id(elements, target_id):
    for element in elements:
        if element["id"] == target_id:
            return element
    return None


def tap_som_target_with_appium(driver, elements, target_id, duration_ms=None):
    element = get_element_by_id(elements, target_id)
    if not element:
        raise ValueError(f"Unknown Set-of-Mark target id: {target_id}")

    x = element["center"]["x"]
    y = element["center"]["y"]
    if duration_ms is None:
        driver.tap([(x, y)])
    else:
        driver.tap([(x, y)], duration_ms)

    return {"x": x, "y": y, "target_id": target_id}


def append_vgt_baseline_record(log_path, record):
    if not log_path:
        return None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(log_path)


def _html_image_data_uri(image_path):
    if not image_path:
        return None

    image_file = Path(image_path)
    if not image_file.exists():
        return None

    suffix = image_file.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _default_som_html_path(marked_image_path):
    marked_path = Path(marked_image_path)
    return marked_path.with_suffix(".html")


def write_som_html_report(som_result, agent_result=None, html_path=None, instruction=None, language=None):
    html_path = Path(html_path or _default_som_html_path(som_result["marked_image_path"]))
    html_path.parent.mkdir(parents=True, exist_ok=True)

    source_image = som_result.get("source_image_path")
    marked_image = som_result.get("marked_image_path")
    source_data_uri = _html_image_data_uri(source_image)
    marked_data_uri = _html_image_data_uri(marked_image)
    target_id = agent_result.get("target_id") if agent_result else None
    selected_element = get_element_by_id(som_result["elements"], target_id) if target_id is not None else None

    def image_panel(title, image_path, data_uri):
        safe_title = html.escape(title)
        if not image_path or not data_uri:
            return f"<section class='panel'><h2>{safe_title}</h2><div class='missing'>Image unavailable</div></section>"
        return (
            f"<section class='panel'><h2>{safe_title}</h2>"
            f"<a href='{html.escape(str(image_path))}' target='_blank'>"
            f"<img src='{data_uri}' alt='{safe_title}' /></a></section>"
        )

    element_rows = []
    for element in som_result["elements"]:
        is_selected = element["id"] == target_id
        bbox = element["bbox"]
        center = element["center"]
        element_rows.append(
            f"""
            <tr class='{"selected" if is_selected else ""}'>
                <td class='num'>[{element['id']}]</td>
                <td class='num'>{center['x']}</td>
                <td class='num'>{center['y']}</td>
                <td class='num'>{bbox['x']}</td>
                <td class='num'>{bbox['y']}</td>
                <td class='num'>{bbox['width']}</td>
                <td class='num'>{bbox['height']}</td>
                <td class='num'>{element['area']}</td>
            </tr>
            """
        )

    if agent_result:
        agent_html = f"""
        <section class='panel agent'>
            <h2>Agent Result</h2>
            <div class='agent-grid'>
                <div><span>Status</span><strong>{html.escape(str(agent_result.get('status')))}</strong></div>
                <div><span>Target ID</span><strong>{html.escape(str(agent_result.get('target_id')))}</strong></div>
                <div><span>Confidence</span><strong>{html.escape(str(agent_result.get('confidence')))}</strong></div>
                <div><span>Model</span><strong>{html.escape(str(agent_result.get('model', 'n/a')))}</strong></div>
            </div>
            <p>{html.escape(str(agent_result.get('rationale') or agent_result.get('reason') or ''))}</p>
        </section>
        """
    else:
        agent_html = """
        <section class='panel agent'>
            <h2>Agent Result</h2>
            <p class='muted'>No instruction was provided, so only Set-of-Mark detection was run.</p>
        </section>
        """

    selected_html = ""
    if selected_element:
        center = selected_element["center"]
        selected_html = (
            f"<p class='selected-callout'>Selected [{selected_element['id']}] "
            f"at center ({center['x']}, {center['y']}).</p>"
        )

    document = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Set-of-Mark VGT Report</title>
    <style>
        :root {{
            --bg: #f6f8fb;
            --panel: #ffffff;
            --text: #172033;
            --muted: #627084;
            --border: #d9e1ea;
            --accent: #d32f2f;
            --accent-soft: #ffebee;
            --shadow: 0 14px 36px rgba(23, 32, 51, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
        .shell {{ max-width: 1320px; margin: 0 auto; padding: 28px 18px 44px; }}
        header {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); padding: 22px; margin-bottom: 18px; }}
        h1 {{ margin: 0 0 10px; font-size: 30px; }}
        h2 {{ margin: 0 0 12px; font-size: 18px; }}
        p {{ color: var(--muted); line-height: 1.5; }}
        .meta {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
        .pill {{ border: 1px solid var(--border); background: #fbfcfe; border-radius: 999px; padding: 6px 10px; font-size: 13px; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; align-items: start; }}
        .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); padding: 16px; overflow: hidden; }}
        .panel img {{ width: 100%; height: auto; display: block; border: 1px solid var(--border); border-radius: 8px; background: white; }}
        .agent {{ margin: 18px 0; }}
        .agent-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
        .agent-grid div {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px; background: #fbfcfe; }}
        .agent-grid span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
        .agent-grid strong {{ font-size: 16px; }}
        .selected-callout {{ color: #b71c1c; background: var(--accent-soft); border: 1px solid #ffcdd2; border-radius: 8px; padding: 10px 12px; }}
        .table-wrap {{ overflow-x: auto; margin-top: 18px; }}
        table {{ width: 100%; border-collapse: collapse; min-width: 760px; background: white; border-radius: 8px; overflow: hidden; }}
        th, td {{ border-bottom: 1px solid var(--border); padding: 9px 10px; text-align: left; }}
        th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; background: #fbfcfe; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        tr.selected td {{ background: var(--accent-soft); font-weight: 800; color: #b71c1c; }}
        .missing {{ min-height: 220px; display: grid; place-items: center; border: 1px dashed var(--border); border-radius: 8px; color: var(--muted); }}
        .muted {{ color: var(--muted); }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class='shell'>
        <header>
            <h1>Set-of-Mark VGT Report</h1>
            <p>Single screenshot UI candidate detection for visual grounding tests.</p>
            {selected_html}
            <div class='meta'>
                <span class='pill'>Elements: {som_result['count']}</span>
                <span class='pill'>Runtime: {som_result.get('time_ms', 0):.1f}ms</span>
                <span class='pill'>Language: {html.escape(str(language or 'n/a'))}</span>
                <span class='pill'>Instruction: {html.escape(str(instruction or 'n/a'))}</span>
            </div>
        </header>
        <div class='grid'>
            {image_panel('Original Screenshot', source_image, source_data_uri)}
            {image_panel('Set-of-Mark Screenshot', marked_image, marked_data_uri)}
        </div>
        {agent_html}
        <section class='panel'>
            <h2>ID to Coordinate Map</h2>
            <div class='table-wrap'>
                <table>
                    <thead><tr><th>ID</th><th>Center X</th><th>Center Y</th><th>Box X</th><th>Box Y</th><th>Width</th><th>Height</th><th>Area</th></tr></thead>
                    <tbody>{''.join(element_rows)}</tbody>
                </table>
            </div>
        </section>
    </div>
</body>
</html>"""

    html_path.write_text(document, encoding="utf-8")
    print(f"📄 Set-of-Mark HTML report saved to: {html_path}")
    return str(html_path)


def _guess_language_from_folder(folder_path):
    folder_name = Path(folder_path).name
    if folder_name.startswith("baseline_"):
        return folder_name.replace("baseline_", "").replace("_", "-")
    return folder_name


def _relative_report_path(path_value, report_root):
    return os.path.relpath(Path(path_value).resolve(), Path(report_root).resolve()).replace(os.sep, "/")


def write_som_batch_report(batch_summary, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    html_path = output_root / "som_summary.html"
    json_path = output_root / "som_summary.json"

    json_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    total_elements = 0
    completed_count = 0
    missing_count = 0
    for folder in batch_summary["folders"]:
        folder_name = html.escape(folder["folder_name"])
        for item in folder["pages"]:
            page_name = html.escape(item["page"])
            if item["status"] != "completed":
                missing_count += 1
                rows.append(
                    f"""
                    <tr>
                        <td>{folder_name}</td>
                        <td>{page_name}</td>
                        <td class='muted'>missing</td>
                        <td class='muted'>-</td>
                        <td class='muted'>-</td>
                    </tr>
                    """
                )
                continue

            completed_count += 1
            total_elements += item["element_count"]
            detail_path = _relative_report_path(item["html_report"], output_root)
            marked_path = _relative_report_path(item["marked_image_path"], output_root)
            source_path = _relative_report_path(item["source_image_path"], output_root)
            agent = item.get("agent_result") or {}
            agent_text = "not run"
            if agent:
                agent_text = f"{agent.get('status')} / target {agent.get('target_id')}"
            rows.append(
                f"""
                <tr>
                    <td>{folder_name}</td>
                    <td>{page_name}</td>
                    <td class='num'>{item['element_count']}</td>
                    <td>{html.escape(agent_text)}</td>
                    <td class='links'>
                        <a href='{detail_path}' target='_blank'>HTML</a>
                        <a href='{marked_path}' target='_blank'>SoM PNG</a>
                        <a href='{source_path}' target='_blank'>Source</a>
                    </td>
                </tr>
                """
            )

    avg_elements = total_elements / completed_count if completed_count else 0
    document = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Set-of-Mark Batch Report</title>
    <style>
        :root {{
            --bg: #f6f8fb;
            --panel: #ffffff;
            --text: #172033;
            --muted: #627084;
            --border: #d9e1ea;
            --accent: #d32f2f;
            --shadow: 0 14px 36px rgba(23, 32, 51, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
        .shell {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 44px; }}
        header, .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }}
        header {{ padding: 22px; margin-bottom: 18px; }}
        h1 {{ margin: 0 0 10px; font-size: 30px; }}
        p {{ color: var(--muted); line-height: 1.5; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }}
        .stat {{ border: 1px solid var(--border); background: #fbfcfe; border-radius: 8px; padding: 12px; }}
        .stat span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
        .stat strong {{ font-size: 22px; }}
        .panel {{ padding: 16px; overflow: hidden; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; min-width: 780px; }}
        th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; }}
        th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; background: #fbfcfe; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .muted {{ color: var(--muted); }}
        .links {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class='shell'>
        <header>
            <h1>Set-of-Mark Batch Report</h1>
            <p>Batch UI candidate detection for visual grounding baseline experiments.</p>
            <div class='stats'>
                <div class='stat'><span>Folders</span><strong>{len(batch_summary['folders'])}</strong></div>
                <div class='stat'><span>Completed pages</span><strong>{completed_count}</strong></div>
                <div class='stat'><span>Missing pages</span><strong>{missing_count}</strong></div>
                <div class='stat'><span>Total elements</span><strong>{total_elements}</strong></div>
                <div class='stat'><span>Avg elements/page</span><strong>{avg_elements:.1f}</strong></div>
            </div>
        </header>
        <section class='panel'>
            <div class='table-wrap'>
                <table>
                    <thead><tr><th>Folder</th><th>Page</th><th>Elements</th><th>Agent</th><th>Artifacts</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </section>
    </div>
</body>
</html>"""
    html_path.write_text(document, encoding="utf-8")
    print(f"📄 Set-of-Mark batch HTML report saved to: {html_path}")
    print(f"🧾 Set-of-Mark batch JSON saved to: {json_path}")
    return {"html_report": str(html_path), "json_report": str(json_path)}


def run_som_batch(
    folders,
    pages,
    output_root,
    instruction=None,
    model=None,
    vgt_log=None,
):
    output_root = Path(output_root) / "som_reports"
    batch_summary = {
        "output_root": str(output_root),
        "folders": [],
        "pages": pages,
        "instruction": instruction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for folder in folders:
        folder_path = Path(folder)
        folder_name = folder_path.name
        language = _guess_language_from_folder(folder_path)
        folder_entry = {"folder_name": folder_name, "folder_path": str(folder_path), "language": language, "pages": []}
        print(f"\n=== Set-of-Mark batch: {folder_name} ===")

        for page_name in pages:
            source_path = folder_path / f"{page_name}.png"
            page_output_dir = output_root / folder_name
            marked_path = page_output_dir / f"{page_name}_som.png"
            detail_html_path = page_output_dir / f"{page_name}_som.html"

            if not source_path.exists():
                print(f"Skipping {page_name}: missing screenshot {source_path}")
                folder_entry["pages"].append({
                    "page": page_name,
                    "source_image_path": str(source_path),
                    "status": "missing",
                })
                continue

            if instruction:
                result = run_visual_grounding_step(
                    image_path=source_path,
                    instruction=instruction,
                    output_path=marked_path,
                    language=language,
                    model=model,
                    log_path=vgt_log,
                )
                html_report = write_som_html_report(
                    som_result=result["som"],
                    agent_result=result["agent"],
                    html_path=detail_html_path,
                    instruction=instruction,
                    language=language,
                )
                som_result = result["som"]
                agent_result = result["agent"]
            else:
                som_result = detect_ui_elements_set_of_mark(source_path, output_path=marked_path)
                agent_result = None
                html_report = write_som_html_report(
                    som_result=som_result,
                    html_path=detail_html_path,
                    language=language,
                )

            page_entry = {
                "page": page_name,
                "source_image_path": som_result["source_image_path"],
                "marked_image_path": som_result["marked_image_path"],
                "html_report": html_report,
                "element_count": som_result["count"],
                "time_ms": som_result["time_ms"],
                "agent_result": agent_result,
                "status": "completed",
            }
            folder_entry["pages"].append(page_entry)

        batch_summary["folders"].append(folder_entry)

    reports = write_som_batch_report(batch_summary, output_root)
    batch_summary.update(reports)
    return batch_summary


def run_visual_grounding_step(
    image_path,
    instruction,
    output_path=None,
    language=None,
    model=None,
    driver=None,
    log_path=None,
):
    som_result = detect_ui_elements_set_of_mark(image_path, output_path=output_path)
    agent_result = choose_som_target_with_ai(
        marked_image_path=som_result["marked_image_path"],
        instruction=instruction,
        elements=som_result["elements"],
        language=language,
        model=model,
    )

    tap_result = None
    if driver is not None and agent_result.get("target_id") is not None:
        tap_result = tap_som_target_with_appium(driver, som_result["elements"], agent_result["target_id"])

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "instruction": instruction,
        "language": language,
        "marked_image_path": som_result["marked_image_path"],
        "element_count": som_result["count"],
        "agent_result": agent_result,
        "tap_result": tap_result,
        "success": tap_result is not None if driver is not None else None,
    }
    append_vgt_baseline_record(log_path, record)
    return {
        "som": som_result,
        "agent": agent_result,
        "tap": tap_result,
        "record": record,
    }


def _find_screenshot_file(folder, page_name, app_name=None):
    """
    Finds a screenshot image in folder matching page_name, handling potential prefixes like katapp_ or bluesky_.
    """
    folder = Path(folder)
    if not folder.exists():
        return None
    candidates = [
        folder / f"{page_name}.png",
    ]
    if app_name:
        candidates.append(folder / f"{app_name}_{page_name}.png")
        if page_name.startswith(f"{app_name}_"):
            candidates.append(folder / f"{page_name[len(app_name)+1:]}.png")
    for prefix in ["bluesky_", "katapp_"]:
        if page_name.startswith(prefix):
            candidates.append(folder / f"{page_name[len(prefix):]}.png")
        else:
            candidates.append(folder / f"{prefix}{page_name}.png")
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Fallback for sequential flow steps: match by step number prefix (e.g. "22_")
    m = re.match(r"^(\w+_)?(\d{2})_", page_name)
    if m:
        step_num = m.group(2)
        patterns = [f"{step_num}_*.png"]
        if app_name:
            patterns.append(f"{app_name}_{step_num}_*.png")
        for prefix in ["bluesky_", "katapp_"]:
            patterns.append(f"{prefix}{step_num}_*.png")
        for pat in patterns:
            matched = sorted(folder.glob(pat))
            if matched:
                return matched[0]

    return None


def get_available_apps():
    """Lists all configured apps based on configs/*.json."""
    configs_dir = Path(__file__).resolve().parent / "configs"
    if not configs_dir.exists():
        return []
    return sorted([f.stem for f in configs_dir.glob("*.json") if not f.stem.startswith("template")])


def resolve_app_config(app_name=None):
    """
    Finds and loads the app config JSON file from configs/.
    Returns: (selected_app_name, config_dict, config_path)
    """
    configs_dir = Path(__file__).resolve().parent / "configs"
    available = {}
    if configs_dir.exists():
        for f in configs_dir.glob("*.json"):
            if not f.stem.startswith("template"):
                available[f.stem.lower()] = f

    selected_app = None
    config_path = None

    if app_name:
        key = str(app_name).lower().replace(".json", "")
        if key in available:
            selected_app = key
            config_path = available[key]
        elif (configs_dir / f"{app_name}.json").exists():
            selected_app = app_name
            config_path = configs_dir / f"{app_name}.json"
        elif Path(app_name).exists():
            selected_app = Path(app_name).stem
            config_path = Path(app_name)
        else:
            matches = [k for k in available if str(app_name).lower() in k]
            if matches:
                selected_app = matches[0]
                config_path = available[selected_app]

    if not config_path:
        if "katapp" in available:
            selected_app = "katapp"
            config_path = available["katapp"]
        elif available:
            selected_app = next(iter(available.keys()))
            config_path = available[selected_app]

    config_data = {}
    if config_path and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load configuration file '{config_path}': {e}")

    return selected_app, config_data, config_path


def build_default_output_path(baseline_path, actual_path):
    baseline_stem = Path(baseline_path).stem
    actual_stem = Path(actual_path).stem
    baseline_parent = Path(baseline_path).parent.name
    actual_parent = Path(actual_path).parent.name
    return Path("./diff_results") / f"diff_{baseline_parent}_{baseline_stem}_vs_{actual_parent}_{actual_stem}.png"


def compare_folder_pair(baseline_folder, actual_folder, output_root, pages, threshold=38, min_box_size=14, use_ai=False, ai_model=None, only_diffs=False, app=None,
                        mask_top=140, mask_bottom=70, ignore_regions=None, ai_backend=None, ai_max_boxes=8):
    baseline_folder = Path(baseline_folder)
    actual_folder = Path(actual_folder)
    output_root = Path(output_root)

    print(f"\n=== Comparing {actual_folder.name} against {baseline_folder.name} ===")

    folder_results = []

    for page_name in pages:
        baseline_path = _find_screenshot_file(baseline_folder, page_name, app_name=app)
        actual_path = _find_screenshot_file(actual_folder, page_name, app_name=app)

        if not baseline_path:
            print(f"Skipping {page_name}: missing baseline file in {baseline_folder}")
            continue

        if not actual_path:
            expected_path = actual_folder / f"{page_name}.png"
            print(f"Skipping {page_name}: missing actual file in {actual_folder}")
            folder_results.append({
                "page": page_name,
                "baseline_path": str(baseline_path),
                "actual_path": str(expected_path),
                "output_path": None,
                "cv_diff_count": None,
                "pixel_diff_count": None,
                "cv_time_ms": None,
                "pixel_time_ms": None,
                "ai_analysis": None,
                "status": "missing_actual",
            })
            continue

        output_path = output_root / actual_folder.name / f"diff_{page_name}.png"
        page_ignore_rects = _ignore_rects_for_page(ignore_regions, actual_folder.name, page_name)
        comparison_result = compare_ui_screenshots(
            baseline_path,
            actual_path,
            output_path,
            threshold=threshold,
            min_box_size=min_box_size,
            ignore_top=mask_top,
            ignore_bottom=mask_bottom,
            ignore_regions=page_ignore_rects,
        )

        cv_count = comparison_result["cv"]["count"]
        pixel_count = comparison_result["pixel"]["count"]

        if only_diffs and cv_count == 0 and pixel_count == 0:
            print(f"   ℹ️ Skipping identical page: {page_name} (0 differences)")
            continue

        ai_analysis = None
        if use_ai and ai_backend == "local":
            from common_agent.diff_triage import triage_page_local
            ai_analysis = triage_page_local(
                baseline_path=baseline_path,
                actual_path=actual_path,
                boxes=comparison_result["cv"]["boxes"],
                page_name=page_name,
                model_path=ai_model,
                max_boxes=ai_max_boxes,
                output_path=output_root / actual_folder.name / f"triage_{page_name}.png",
            )
        elif use_ai:
            ai_analysis = analyze_diff_with_ai(
                baseline_path=baseline_path,
                actual_path=actual_path,
                cv_diff_path=comparison_result["cv"]["output_path"],
                page_name=page_name,
                cv_diff_count=cv_count,
                model=ai_model,
            )

        page_result = {
            "page": page_name,
            "baseline_path": str(baseline_path),
            "actual_path": str(actual_path),
            "output_path": comparison_result["cv"]["output_path"],
            "pixel_output_path": comparison_result["pixel"]["output_path"],
            "cv_diff_count": cv_count,
            "pixel_diff_count": pixel_count,
            "cv_time_ms": comparison_result["cv"]["time_ms"],
            "pixel_time_ms": comparison_result["pixel"]["time_ms"],
            "ai_analysis": ai_analysis,
            "status": "compared",
        }
        if page_ignore_rects:
            page_result["ignored_regions"] = [list(rect) for rect in page_ignore_rects]
        if ai_backend == "local":
            page_result["cv_boxes"] = [list(box) for box in comparison_result["cv"]["boxes"]]
        folder_results.append(page_result)

    return folder_results


def run_batch_comparison(baseline_folder, actual_folders, pages, output_root, threshold=38, min_box_size=14, use_ai=False, ai_model=None, only_diffs=False, app=None, compare_options=None):
    batch_start_time = time.time()
    all_results = []
    for actual_folder in actual_folders:
        folder_results = compare_folder_pair(
            baseline_folder=baseline_folder,
            actual_folder=actual_folder,
            output_root=output_root,
            pages=pages,
            threshold=threshold,
            min_box_size=min_box_size,
            use_ai=use_ai,
            ai_model=ai_model,
            only_diffs=only_diffs,
            app=app,
            **(compare_options or {}),
        )
        all_results.append({
            "actual_folder": str(actual_folder),
            "results": folder_results,
        })

    write_batch_report(
        baseline_folder=baseline_folder,
        actual_folders=actual_folders,
        pages=pages,
        output_root=output_root,
        batch_results=all_results,
        threshold=threshold,
        min_box_size=min_box_size,
        use_ai=use_ai,
        ai_model=ai_model,
        total_runtime_ms=(time.time() - batch_start_time) * 1000,
        app=app,
        compare_options=compare_options,
    )


def run_version_comparison(folder_pairs, pages, output_root, baseline_root, threshold=38, min_box_size=14, use_ai=False, ai_model=None, only_diffs=False, app=None, baseline_label=None, actual_label=None, compare_options=None):
    """Compares each language folder of the previous release with the same folder of the current release."""
    batch_start_time = time.time()
    all_results = []
    for baseline_folder, actual_folder in folder_pairs:
        folder_results = compare_folder_pair(
            baseline_folder=baseline_folder,
            actual_folder=actual_folder,
            output_root=output_root,
            pages=pages,
            threshold=threshold,
            min_box_size=min_box_size,
            use_ai=use_ai,
            ai_model=ai_model,
            only_diffs=only_diffs,
            app=app,
            **(compare_options or {}),
        )
        all_results.append({
            "actual_folder": str(actual_folder),
            "results": folder_results,
        })

    write_batch_report(
        baseline_folder=baseline_root,
        actual_folders=[actual for _, actual in folder_pairs],
        pages=pages,
        output_root=output_root,
        batch_results=all_results,
        threshold=threshold,
        min_box_size=min_box_size,
        use_ai=use_ai,
        ai_model=ai_model,
        total_runtime_ms=(time.time() - batch_start_time) * 1000,
        app=app,
        comparison_mode="version",
        baseline_label=baseline_label,
        actual_label=actual_label,
        compare_options=compare_options,
    )


def _build_folder_summaries(batch_results):
    folder_summaries = []
    for folder_entry in batch_results:
        folder_name = Path(folder_entry["actual_folder"]).name
        folder_results = folder_entry["results"]
        compared_pages = [item for item in folder_results if item["status"] == "compared"]
        missing_pages = [item for item in folder_results if item["status"] != "compared"]
        
        cv_total = sum(item.get("cv_diff_count") or 0 for item in compared_pages)
        pixel_total = sum(item.get("pixel_diff_count") or 0 for item in compared_pages)
        cv_time_total = sum(item.get("cv_time_ms") or 0 for item in compared_pages)
        pixel_time_total = sum(item.get("pixel_time_ms") or 0 for item in compared_pages)

        folder_summaries.append({
            "folder_name": folder_name,
            "actual_folder": folder_entry["actual_folder"],
            "results": folder_results,
            "compared_count": len(compared_pages),
            "missing_count": len(missing_pages),
            "cv_difference_total": cv_total,
            "pixel_difference_total": pixel_total,
            "cv_time_total_ms": cv_time_total,
            "pixel_time_total_ms": pixel_time_total,
        })

    return folder_summaries


def _build_page_rankings(batch_results):
    page_totals = {}

    for folder_entry in batch_results:
        folder_name = Path(folder_entry["actual_folder"]).name
        for item in folder_entry["results"]:
            if item["status"] != "compared":
                continue

            page_name = item["page"]
            entry = page_totals.setdefault(page_name, {
                "page": page_name,
                "cv_total_differences": 0,
                "pixel_total_differences": 0,
                "cv_total_time_ms": 0,
                "pixel_total_time_ms": 0,
                "comparisons": 0,
                "by_folder": [],
            })
            entry["cv_total_differences"] += item.get("cv_diff_count") or 0
            entry["pixel_total_differences"] += item.get("pixel_diff_count") or 0
            entry["cv_total_time_ms"] += item.get("cv_time_ms") or 0
            entry["pixel_total_time_ms"] += item.get("pixel_time_ms") or 0
            entry["comparisons"] += 1
            entry["by_folder"].append({
                "folder_name": folder_name,
                "cv_diff_count": item.get("cv_diff_count") or 0,
                "pixel_diff_count": item.get("pixel_diff_count") or 0,
                "output_path": item["output_path"],
            })

    rankings = sorted(
        page_totals.values(),
        key=lambda item: (-item["cv_total_differences"], item["page"]),
    )

    for index, item in enumerate(rankings, start=1):
        item["rank"] = index

    return rankings


def write_batch_report(baseline_folder, actual_folders, pages, output_root, batch_results, threshold, min_box_size, use_ai=False, ai_model=None, total_runtime_ms=None, app=None, comparison_mode=None, baseline_label=None, actual_label=None, compare_options=None):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    folder_summaries = _build_folder_summaries(batch_results)
    page_rankings = _build_page_rankings(batch_results)

    summary = {
        "app": app,
        "comparison_mode": comparison_mode,
        "baseline_label": baseline_label,
        "actual_label": actual_label,
        "output_root": str(output_root),
        "baseline_folder": str(Path(baseline_folder)),
        "actual_folders": [str(Path(folder)) for folder in actual_folders],
        "pages": pages,
        "threshold": threshold,
        "min_box_size": min_box_size,
        "ai_enabled": use_ai,
        "ai_model": ai_model or os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        "comparisons": batch_results,
        "folder_summaries": folder_summaries,
        "page_rankings": page_rankings,
        "total_runtime_ms": total_runtime_ms,
    }
    if compare_options:
        summary["compare_options"] = compare_options

    json_path = output_root / "summary_report.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)

    html_path = output_root / "summary_report.html"
    with html_path.open("w", encoding="utf-8") as html_file:
        html_file.write(build_html_report(summary))

    print(f"\n📄 HTML report saved to: {html_path}")
    print(f"🧾 Summary JSON saved to: {json_path}")


def build_html_report(summary):
    baseline_folder = html.escape(summary["baseline_folder"])
    actual_folders = summary["folder_summaries"]
    page_rankings = summary["page_rankings"]
    total_compared = sum(folder["compared_count"] for folder in actual_folders)
    total_missing = sum(folder["missing_count"] for folder in actual_folders)
    cv_total_differences = sum(folder["cv_difference_total"] for folder in actual_folders)
    pixel_total_differences = sum(folder["pixel_difference_total"] for folder in actual_folders)
    total_runtime_ms = summary.get("total_runtime_ms")
    ai_enabled = summary.get("ai_enabled", False)
    ai_items = [
        item.get("ai_analysis")
        for folder in actual_folders
        for item in folder["results"]
        if item.get("ai_analysis")
    ]
    ai_fail_count = sum(1 for item in ai_items if item.get("severity") == "fail")
    ai_warn_count = sum(1 for item in ai_items if item.get("severity") == "warn")
    ai_pass_count = sum(1 for item in ai_items if item.get("severity") == "pass")
    report_root = Path(summary.get("output_root", "./diff_results")).resolve()
    app_name = summary.get("app")
    app_eyebrow = f" · {app_name.upper()}" if app_name else ""
    app_heading = f"{app_name.capitalize()} " if app_name else ""
    baseline_name = Path(summary["baseline_folder"]).name
    actual_paths = summary.get("actual_folders") or []
    is_version_comparison = summary.get("comparison_mode") == "version" or (
        bool(actual_paths) and all(Path(p).name == baseline_name for p in actual_paths)
    )
    if is_version_comparison:
        report_title = "Version comparison"
        report_subtitle = ("This report compares the screenshots of the current build against a previous build "
                           "in the same language, then highlights the pages with the most layout drift.")
        folders_meta = "Language folders compared with the previous build"
        build_labels = []
        if summary.get("baseline_label"):
            build_labels.append(f"Baseline build: <strong>{html.escape(str(summary['baseline_label']))}</strong>.")
        if summary.get("actual_label"):
            build_labels.append(f"Current build: <strong>{html.escape(str(summary['actual_label']))}</strong>.")
        if build_labels:
            report_subtitle += " " + " ".join(build_labels)
    else:
        report_title = "Language screenshot comparison"
        report_subtitle = ("This report compares reference screenshots against secondary language sets, "
                           "then highlights the pages with the most layout drift.")
        folders_meta = "Language sets compared"

    def rel_path(path_value):
        return os.path.relpath(Path(path_value).resolve(), report_root).replace(os.sep, "/")

    def image_data_uri(image_path):
        if not image_path:
            return None

        image_file = Path(image_path).resolve()
        if not image_file.exists():
            return None

        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def image_cell(file_path, label):
        if not file_path:
            return f"<div class='image-missing'>{html.escape(label)} unavailable</div>"

        safe_path = rel_path(file_path)
        data_uri = image_data_uri(file_path)
        safe_label = html.escape(label)
        if data_uri:
            return f"<a class='image-link' href='{safe_path}' target='_blank' title='{safe_label}'><img src='{data_uri}' alt='{safe_label}' loading='lazy' /></a>"
        return f"<a class='image-link' href='{safe_path}' target='_blank' title='{safe_label}'><div class='image-missing'>{safe_label} unavailable</div></a>"

    def ai_cell(ai_analysis):
        if not ai_enabled:
            return "<span class='muted'>disabled</span>"
        if not ai_analysis:
            return "<span class='muted'>not run</span>"

        status = html.escape(ai_analysis.get("status", "unknown"))
        severity = html.escape(ai_analysis.get("severity") or status)
        summary_text = html.escape(ai_analysis.get("summary") or ai_analysis.get("reason") or "")
        actions = ai_analysis.get("recommended_actions") or []
        action_html = ""
        if actions:
            action_html = "<ul class='compact-list ai-actions'>" + "".join(
                f"<li>{html.escape(str(action))}</li>" for action in actions[:3]
            ) + "</ul>"
        triage_html = ""
        if ai_analysis.get("triage_image"):
            triage_html = f"<p><a href='{html.escape(rel_path(ai_analysis['triage_image']))}' target='_blank'>rated boxes (local VLM)</a></p>"
        return f"<div class='ai-box'><span class='ai-pill ai-{severity}'>{severity}</span><p>{summary_text}</p>{action_html}{triage_html}</div>"

    def page_row(item):
        page = html.escape(item["page"])
        if item["status"] == "compared" and item["output_path"]:
            cv_count = item.get("cv_diff_count", 0) or 0
            pixel_count = item.get("pixel_diff_count", 0) or 0
            cv_time = item.get("cv_time_ms", 0) or 0
            pixel_time = item.get("pixel_time_ms", 0) or 0
            is_identical = (cv_count == 0 and pixel_count == 0)
            row_class = "row-identical" if is_identical else "row-different"
            diff_attr = "0" if is_identical else "1"
            status_badge = "<span class='badge-identical'>✅ 0 Diffs</span>" if is_identical else ""
            return f"""
                <tr class='{row_class}' data-diff='{diff_attr}'>
                    <td class='page-name'>{page} {status_badge}</td>
                    <td class='num'><span class='cv-badge'>CV: {cv_count}/{cv_time:.1f}ms</span></td>
                    <td class='num'><span class='pixel-badge'>Pixel: {pixel_count}/{pixel_time:.1f}ms</span></td>
                    <td>{ai_cell(item.get('ai_analysis'))}</td>
                    <td class='thumb-cell'>{image_cell(item['baseline_path'], f'{page} baseline')}</td>
                    <td class='thumb-cell'>{image_cell(item['actual_path'], f'{page} actual')}</td>
                    <td class='thumb-cell'>{image_cell(item['output_path'], f'{page} CV diff')}</td>
                    <td class='thumb-cell'>{image_cell(item.get('pixel_output_path'), f'{page} Pixel diff')}</td>
                </tr>
            """

        return f"""
            <tr class='row-missing' data-diff='0'>
                <td class='page-name'>{page}</td>
                <td class='num muted'>missing</td>
                <td class='num muted'>-</td>
                <td class='num muted'>-</td>
                <td class='num muted'>-</td>
                <td class='thumb-cell'><div class='image-missing'>baseline unavailable</div></td>
                <td class='thumb-cell'><div class='image-missing'>actual unavailable</div></td>
                <td class='thumb-cell'><div class='image-missing'>diff unavailable</div></td>
                <td class='thumb-cell'><div class='image-missing'>pixel diff unavailable</div></td>
            </tr>
        """

    overview_sections = []
    for folder in actual_folders:
        rows = [page_row(item) for item in folder["results"]]

        overview_sections.append(
           f"""
           <section class='card'>
               <div class='card-header'>
                   <div>
                       <h2>{html.escape(folder['folder_name'])}</h2>
                       <p>Compared pages: {folder['compared_count']} · Missing pages: {folder['missing_count']}</p>
                       <p><small>CV Grayscale: {folder['cv_difference_total']} diffs ({folder['cv_time_total_ms']:.1f}ms) | Pixel Matching: {folder['pixel_difference_total']} diffs ({folder['pixel_time_total_ms']:.1f}ms)</small></p>
                   </div>
                   <div class='filter-bar'>
                       <button class='filter-btn active' onclick="filterTableRows('all', this)">All</button>
                       <button class='filter-btn' onclick="filterTableRows('diff', this)">Only Differences</button>
                       <button class='filter-btn' onclick="filterTableRows('identical', this)">Identical (0 Diffs)</button>
                   </div>
               </div>
               <div class='table-wrap'>
                   <table>
                       <thead><tr><th>Page</th><th>CV Diffs/Time</th><th>Pixel Diffs/Time</th><th>AI</th><th>Baseline</th><th>Actual</th><th>Diff CV</th><th>Diff Pixel</th></tr></thead>
                       <tbody>{''.join(rows)}</tbody>
                   </table>
               </div>
           </section>
           """
        )

    ranking_cards = []
    ranking_rows = []
    for item in page_rankings:
        folder_links = []
        for compare in item["by_folder"]:
            diff_path = rel_path(compare["output_path"])
            cv_count = compare.get("cv_diff_count", 0)
            pixel_count = compare.get("pixel_diff_count", 0)
            folder_links.append(
                f"<li><a href='{diff_path}' target='_blank'>{html.escape(compare['folder_name'])}</a>: CV={cv_count} | Pixel={pixel_count}</li>"
            )

        ranking_cards.append(
            f"""
            <article class='ranking-card'>
                <div class='rank-badge'>#{item['rank']}</div>
                <div>
                    <p class='title'>{html.escape(item['page'])}</p>
                    <p class='meta'>Compared in {item['comparisons']} folders</p>
                    <ul class='compact-list'>
                        {''.join(folder_links)}
                    </ul>
                </div>
                <div class='score'><small>CV: {item['cv_total_differences']}</small><br/><small>Pixel: {item['pixel_total_differences']}</small></div>
            </article>
            """
        )

        ranking_rows.append(
            f"""
            <tr>
                <td class='num rank'>{item['rank']}</td>
                <td>{html.escape(item['page'])}</td>
                <td class='num'><span class='cv-badge'>CV: {item['cv_total_differences']}</span></td>
                <td class='num'><span class='pixel-badge'>Pixel: {item['pixel_total_differences']}</span></td>
                <td class='num'>{item['comparisons']}</td>
                <td><ul class='compact-list'>{''.join(folder_links)}</ul></td>
            </tr>
            """
        )

    template = Template("""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>${app_heading}Visual Regression Report</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f4f7fb;
            --panel: #ffffff;
            --text: #172033;
            --muted: #5c6b80;
            --border: #d8e0ea;
            --accent: #0f62fe;
            --accent-soft: #eaf1ff;
            --danger: #c62828;
            --shadow: 0 18px 50px rgba(23, 32, 51, 0.08);
            --radius: 18px;
        }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background: linear-gradient(180deg, #eef3fb 0%, #f8fafc 100%); color: var(--text); }
        .shell { max-width: 1400px; margin: 0 auto; padding: 32px 20px 48px; }
        .hero { background: var(--panel); border: 1px solid var(--border); border-radius: 28px; padding: 28px; box-shadow: var(--shadow); margin-bottom: 24px; }
        .eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px; color: var(--accent); font-weight: 700; }
        h1 { margin: 10px 0 12px; font-size: clamp(28px, 4vw, 44px); line-height: 1.05; }
        .subtitle { margin: 0; color: var(--muted); max-width: 900px; line-height: 1.6; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-top: 22px; align-items: center; }
        .stat { background: linear-gradient(180deg, #fff, #f6f9ff); border: 1px solid var(--border); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 6px; }
        .stat .label { color: var(--muted); font-size: 12px; margin: 0; }
        .stat .value { font-size: 22px; font-weight: 800; }
        .stat .meta { color: var(--muted); font-size: 12px; }
        .tabs { display: inline-flex; gap: 8px; background: rgba(255,255,255,0.75); border: 1px solid var(--border); border-radius: 999px; padding: 6px; margin-bottom: 18px; backdrop-filter: blur(8px); position: sticky; top: 12px; z-index: 5; box-shadow: var(--shadow); }
        .tab-btn { border: 0; background: transparent; color: var(--muted); padding: 10px 16px; border-radius: 999px; cursor: pointer; font-weight: 700; }
        .tab-btn.active { background: var(--accent); color: white; }
        .page { display: none; animation: fadeIn 180ms ease-out; }
        .page.active { display: block; }
        .grid { display: grid; gap: 18px; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); overflow: hidden; }
        .card-header { padding: 16px 18px; display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
        .card h2 { margin: 0; font-size: 20px; }
        .card p { margin: 6px 0 0; color: var(--muted); }
        .table-wrap { overflow-x: auto; padding: 14px 16px 18px; }
        table { width: 100%; border-collapse: collapse; min-width: 1000px; background: white; border-radius: 8px; overflow: hidden; }
        thead th { background: linear-gradient(180deg, #fbfdff, #f6f9ff); position: sticky; top: 0; z-index: 2; }
        th, td { border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; vertical-align: middle; }
        th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
        tbody tr { transition: background 120ms ease-in-out; }
        tbody tr:nth-child(odd) td { background: #ffffff; }
        tbody tr:nth-child(even) td { background: #fbfcfe; }
        tr:hover td { background: #eef6ff; }
        .num { text-align: right; font-variant-numeric: tabular-nums; }
        .rank { font-weight: 800; color: var(--accent); }
        .muted { color: var(--muted); }
        .page-name { min-width: 140px; font-weight: 700; }
        .thumb-cell { min-width: 260px; }
        .thumb-row { display: flex; gap: 8px; align-items: center; }
        .thumb-row img { width: 120px; height: auto; border-radius: 10px; border: 1px solid var(--border); }
        .compact-list { margin: 0; padding-left: 18px; }
        .compact-list li { margin: 4px 0; }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }
        .image-link { display: block; }
        .image-link img { display: block; width: 100%; max-width: 220px; height: auto; border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 6px 24px rgba(23, 32, 51, 0.06); background: white; }
        .image-missing { min-height: 100px; display: grid; place-items: center; border: 1px dashed var(--border); border-radius: 10px; color: var(--muted); background: #fbfcfe; padding: 12px; text-align: center; }
        .legend { display:flex; gap:8px; align-items:center; margin-top:10px; }
        .legend .item { display:inline-flex; align-items:center; gap:8px; font-size:13px; color:var(--muted); }
        .legend .swatch { width:18px; height:12px; border-radius:4px; display:inline-block; }
        .legend .cv { background: rgba(255,99,132,0.9); }
        .legend .pixel { background: rgba(34,197,94,0.9); }
        .ranking-list { display: grid; gap: 12px; }
        .ranking-card { display: grid; grid-template-columns: 70px 1fr 140px; gap: 14px; align-items: center; padding: 14px 16px; border: 1px solid var(--border); border-radius: 12px; background: var(--panel); box-shadow: var(--shadow); }
        .ranking-card .rank-badge { width: 56px; height: 56px; border-radius: 12px; display: grid; place-items: center; background: var(--accent-soft); color: var(--accent); font-size: 20px; font-weight: 800; }
        .ranking-card .title { font-size: 16px; font-weight: 800; margin: 0 0 4px; }
        .ranking-card .meta { margin: 0; color: var(--muted); }
        .ranking-card .score { font-size: 16px; font-weight: 800; color: var(--danger); text-align: right; }
        .pill { display: inline-flex; align-items: center; gap: 6px; background: var(--accent-soft); color: var(--accent); border-radius: 999px; padding: 6px 12px; font-size: 13px; font-weight: 700; margin-right: 8px; }
        .cv-badge { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 4px 8px; border-radius: 8px; font-size: 12px; font-weight: 600; }
        .pixel-badge { display: inline-block; background: #f3e5f5; color: #7b1fa2; padding: 4px 8px; border-radius: 8px; font-size: 12px; font-weight: 600; }
        .ai-box { min-width: 220px; max-width: 320px; }
        .ai-box p { margin: 6px 0 0; color: var(--text); font-size: 13px; line-height: 1.4; }
        .ai-actions { margin-top: 6px; color: var(--muted); font-size: 12px; }
        .ai-pill { display: inline-block; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 800; text-transform: uppercase; }
        .ai-pass { background: #e8f5e9; color: #1b5e20; }
        .ai-warn { background: #fff8e1; color: #8a5a00; }
        .ai-fail { background: #ffebee; color: #b71c1c; }
        .ai-unknown, .ai-error { background: #eceff1; color: #455a64; }
        .badge-identical { display: inline-block; background: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; margin-left: 6px; }
        .filter-bar { display: inline-flex; gap: 6px; background: #f0f2f5; padding: 4px; border-radius: 8px; }
        .filter-btn { border: none; background: transparent; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; color: var(--muted); }
        .filter-btn.active { background: white; color: var(--text); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .section-title { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 0 0 14px; }
        .section-title h2 { margin: 0; font-size: 20px; }
        .section-title p { margin: 4px 0 0; color: var(--muted); }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 960px) {
            .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .ranking-card { grid-template-columns: 56px 1fr; }
            .ranking-card .score { grid-column: 2; text-align: left; }
        }
        @media (max-width: 640px) {
            .shell { padding: 16px 12px 32px; }
            .stats { grid-template-columns: 1fr; }
            .tabs { width: 100%; justify-content: space-between; }
            .tab-btn { flex: 1; }
        }
    </style>
</head>
<body>
    <div class='shell'>
        <header class='hero'>
            <div class='eyebrow'>Visual Regression Report${app_eyebrow}</div>
            <h1>${app_heading}${report_title}</h1>
            <p class='subtitle'>Baseline: <strong>$baseline_folder</strong>. $report_subtitle</p>
            <div class='stats'>
                <div class='stat'><div class='label'>Compared folders</div><div class='value'>$compared_folders</div><div class='meta'>$folders_meta</div></div>
                <div class='stat'><div class='label'>Compared pages</div><div class='value'>$total_compared</div><div class='meta'>Total page comparisons</div></div>
                <div class='stat'><div class='label'>CV Grayscale diffs</div><div class='value'>$cv_total_differences</div><div class='meta'>Detected by grayscale method</div></div>
                <div class='stat'><div class='label'>Pixel Matching diffs</div><div class='value'>$pixel_total_differences</div><div class='meta'>Detected by strict pixel comparison</div></div>
                <div class='stat'><div class='label'>Total runtime</div><div class='value'>$total_runtime</div><div class='meta'>End-to-end batch duration</div></div>
                <div class='stat'><div class='label'>Missing pages</div><div class='value'>$total_missing</div><div class='meta'>Pages not found in actual folders</div></div>
                <div class='stat'><div class='label'>AI verdicts</div><div class='value'>$ai_verdicts</div><div class='meta'>$ai_status</div></div>
            </div>
            <div class='legend'>
                <div class='item'><span class='swatch cv'></span> CV: Grayscale diff boxes (red)</div>
                <div class='item'><span class='swatch pixel'></span> Pixel: Exact pixel diff boxes (green)</div>
            </div>
        </header>

        <nav class='tabs' aria-label='Report pages'>
            <button class='tab-btn active' data-target='overview-page'>Overview</button>
            <button class='tab-btn' data-target='ranking-page'>Ranking</button>
        </nav>

        <main>
            <section id='overview-page' class='page active'>
                <div class='grid'>
                    $overview_sections
                </div>
            </section>

            <section id='ranking-page' class='page'>
                <div class='section-title'>
                    <div>
                        <h2>Pages with the most differences</h2>
                        <p>Sorted by the total number of diff boxes across all non-DE language sets.</p>
                    </div>
                    <div>
                        <span class='pill'>Top pages only</span>
                    </div>
                </div>

                <div class='ranking-list'>
                    $ranking_cards
                </div>

                <div class='card' style='margin-top: 18px;'>
                    <div class='card-header'>
                        <div>
                            <h2>Ranking table</h2>
                            <p>Compact table view for the most problematic pages.</p>
                        </div>
                    </div>
                    <div class='table-wrap'>
                        <table>
                            <thead><tr><th>Rank</th><th>Page</th><th>CV Diffs</th><th>Pixel Diffs</th><th>Compared folders</th><th>Per-folder breakdown</th></tr></thead>
                            <tbody>$ranking_rows</tbody>
                        </table>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script>
        const buttons = Array.from(document.querySelectorAll('.tab-btn'));
        const pages = Array.from(document.querySelectorAll('.page'));

        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                buttons.forEach((item) => item.classList.remove('active'));
                pages.forEach((page) => page.classList.remove('active'));
                button.classList.add('active');
                document.getElementById(button.dataset.target).classList.add('active');
            });
        });

        function filterTableRows(type, btn) {
            const card = btn.closest('.card') || document;
            card.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            card.querySelectorAll('tbody tr').forEach(tr => {
                if (type === 'all') tr.style.display = '';
                else if (type === 'diff') tr.style.display = tr.getAttribute('data-diff') === '1' ? '' : 'none';
                else if (type === 'identical') tr.style.display = tr.getAttribute('data-diff') === '0' ? '' : 'none';
            });
        }
    </script>
</body>
</html>""")

    return template.substitute(
        baseline_folder=baseline_folder,
        compared_folders=len(actual_folders),
        total_compared=total_compared,
        total_missing=total_missing,
        cv_total_differences=cv_total_differences,
        pixel_total_differences=pixel_total_differences,
        ai_verdicts=(
            f"{ai_fail_count} fail / {ai_warn_count} warn / {ai_pass_count} pass"
            if ai_enabled else "disabled"
        ),
        ai_status=(
            f"Model: {html.escape(summary.get('ai_model') or 'default')}"
            if ai_enabled else "Run with --ai to enable"
        ),
        total_runtime=(
            f"{total_runtime_ms/1000:.2f}s" if total_runtime_ms is not None else "n/a"
        ),
        app_eyebrow=app_eyebrow,
        app_heading=app_heading,
        report_title=report_title,
        report_subtitle=report_subtitle,
        folders_meta=folders_meta,
        overview_sections=''.join(overview_sections),
        ranking_cards=''.join(ranking_cards),
        ranking_rows=''.join(ranking_rows),
    )


def main():
    parser = argparse.ArgumentParser(description="Compare UI screenshots and mark visual differences with red boxes.")
    parser.add_argument("baseline", nargs="?", default=None, help="Path to the baseline screenshot")
    parser.add_argument("actual", nargs="?", default=None, help="Path to the screenshot to compare against the baseline")
    parser.add_argument("--batch", action="store_true", help="Compare the DE baseline against all pages in all other language folders")
    parser.add_argument("--som-batch", action="store_true", help="Create Set-of-Mark reports for every configured folder/page")
    parser.add_argument("--som-folders", nargs="*", default=None, help="Screenshot folders for Set-of-Mark batch mode. Defaults to baseline folder plus actual folders")
    parser.add_argument("--som", default=None, help="Create a Set-of-Mark image for a single screenshot")
    parser.add_argument("--som-html", default=None, help="Optional HTML report path for Set-of-Mark output")
    parser.add_argument("--instruction", default=None, help="Natural-language target instruction for the visual grounding agent")
    parser.add_argument("--language", default=None, help="Language or locale hint for the visual grounding agent")
    parser.add_argument("--agent-model", default=None, help=f"Vision agent model. Defaults to OPENAI_AGENT_MODEL or {DEFAULT_AGENT_MODEL}")
    parser.add_argument("--vgt-log", default=None, help="Optional JSONL file for baseline VGT experiment records")
    parser.add_argument("--app", "-A", default=None, help="Application name to compare (e.g. 'katapp', 'bluesky'). Reads configuration from configs/<app>.json")
    parser.add_argument("--model", "-m", default=None, help="Model directory containing screenshots (e.g. Qwen_trained, openAI_6.0_Luna, claude)")
    parser.add_argument("--baseline-folder", default=None, help="Baseline screenshot folder for batch mode")
    parser.add_argument("--actual-folders", nargs="*", default=None, help="Actual screenshot folders for batch mode")
    parser.add_argument("--version-baseline-root", default=None, help="Screenshot root of the previous release. With --batch, each language folder is compared with the folder of the same name in the current release (version comparison)")
    parser.add_argument("--version-actual-root", default=None, help="Screenshot root of the current release for the version comparison. Defaults to the resolved screenshots directory")
    parser.add_argument("--baseline-label", default=None, help="Identity of the baseline app build shown in the version comparison report (e.g. commit SHA)")
    parser.add_argument("--actual-label", default=None, help="Identity of the current app build shown in the version comparison report (e.g. commit SHA)")
    parser.add_argument("--pages", nargs="*", default=None, help="Page names to compare in batch mode. Defaults to all matching standard pages. Use '--flow' or '--pages flow' for full step flow.")
    parser.add_argument("--flow", action="store_true", help="Compare all numbered sequential flow step screenshots (01_*, 02_*, etc.)")
    parser.add_argument("--output-root", default="./diff_results", help="Root directory for batch diff images")
    parser.add_argument("--output", default=None, help="Path to save the annotated diff image")
    parser.add_argument("--threshold", type=int, default=38, help="Pixel difference threshold")
    parser.add_argument("--min-box-size", type=int, default=14, help="Minimum width/height for a difference box")
    parser.add_argument("--only-diffs", action="store_true", help="Only compare and report pages with visual differences, ignore identical pages")
    parser.add_argument("--ai", action="store_true", help="Use AI vision to triage CV differences and add verdicts to the report")
    parser.add_argument("--ai-local", action="store_true", help="Rate every difference box with a local VLM (default Qwen/Qwen3-VL-4B-Instruct, override with --ai-model or TRIAGE_MODEL_PATH). Screenshots stay on this machine")
    parser.add_argument("--ai-max-boxes", type=int, default=8, help="Maximum number of difference boxes per page rated by --ai-local (largest first)")
    parser.add_argument("--mask-top", type=int, default=None, help="Masked rows at the top (status bar). Defaults to compare.mask_top of the app config, otherwise 140")
    parser.add_argument("--mask-bottom", type=int, default=None, help="Masked rows at the bottom (home indicator). Defaults to compare.mask_bottom of the app config, otherwise 70")
    parser.add_argument("--ai-model", default=None, help="Vision-capable OpenAI model. Defaults to OPENAI_VISION_MODEL or gpt-4.1-mini")
    args = parser.parse_args()

    # Discover and resolve application config
    app_name, app_config, app_config_path = resolve_app_config(args.app)
    available_apps = get_available_apps()
    if args.app:
        print(f"📱 Selected App: {app_name} (loaded from {app_config_path})")
    else:
        apps_str = ", ".join(available_apps) if available_apps else "none"
        print(f"ℹ️ No --app specified, defaulting to '{app_name}'. Available apps: [{apps_str}]")

    cfg_flows = app_config.get("test_flows", []) if app_config else []
    cfg_folders = [f["folder"] for f in cfg_flows if "folder" in f]

    cmp_cfg = (app_config or {}).get("compare", {})
    compare_options = {}
    for key, cli_value in (("mask_top", args.mask_top), ("mask_bottom", args.mask_bottom)):
        value = cli_value if cli_value is not None else cmp_cfg.get(key)
        if value is not None:
            compare_options[key] = int(value)
    if cmp_cfg.get("ignore_regions"):
        compare_options["ignore_regions"] = cmp_cfg["ignore_regions"]
    use_ai = args.ai or args.ai_local
    ai_model = args.ai_model
    if args.ai_local:
        from common_agent.diff_triage import default_triage_model
        ai_model = ai_model or default_triage_model()
        compare_options["ai_backend"] = "local"
        compare_options["ai_max_boxes"] = args.ai_max_boxes

    # Isolate diff output directory per app (KatApp preserves legacy root ./diff_results)
    output_root = Path(args.output_root)
    if output_root == Path("./diff_results") and app_name and app_name.lower() != "katapp":
        output_root = output_root / app_name

    # Intelligent fallback for screenshots directory
    default_base_dir = Path("./screenshots")
    if args.model:
        model_dir = Path(args.model)
        if not model_dir.exists():
            matched = [d for d in Path(".").iterdir() if d.is_dir() and args.model.lower() in d.name.lower()]
            if matched:
                model_dir = matched[0]
        base_dir = model_dir / "screenshots"
    elif default_base_dir.exists():
        base_dir = default_base_dir
    else:
        base_dir = None
        for candidate in ["Qwen_trained", "openAI_6.0_Luna", "openAI_6.0_Sol", "claude", "Qwen_untrain"]:
            candidate_base = Path(candidate) / "screenshots"
            if candidate_base.exists():
                base_dir = candidate_base
                print(f"ℹ️ Default path './screenshots' not found. Using screenshots from model directory '{candidate}'.")
                break
        if not base_dir:
            base_dir = default_base_dir

    # Resolve baseline_folder
    if args.baseline_folder:
        baseline_folder = Path(args.baseline_folder)
    else:
        baseline_name = cfg_folders[0] if cfg_folders else ("baseline_DE_DE" if app_name == "katapp" else "baseline_EN")
        baseline_folder = base_dir / baseline_name
        if not baseline_folder.exists() and args.baseline is None:
            for candidate in ["Qwen_trained", "openAI_6.0_Luna", "openAI_6.0_Sol", "claude", "Qwen_untrain"]:
                cand_folder = Path(candidate) / "screenshots" / baseline_name
                if cand_folder.exists():
                    baseline_folder = cand_folder
                    base_dir = Path(candidate) / "screenshots"
                    break

    # Resolve actual_folders
    if args.actual_folders:
        actual_folders = [Path(f) for f in args.actual_folders]
    else:
        if len(cfg_folders) > 1:
            actual_names = cfg_folders[1:]
        else:
            actual_names = ["baseline_EN_US", "baseline_ZH_CN", "baseline_ZH_TW"] if app_name == "katapp" else ["baseline_DE", "baseline_ZH_CN"]
        actual_folders = [base_dir / name for name in actual_names]

    if args.som_batch:
        som_folders = args.som_folders or [baseline_folder, *actual_folders]
        result = run_som_batch(
            folders=som_folders,
            pages=args.pages,
            output_root=output_root,
            instruction=args.instruction,
            model=args.agent_model,
            vgt_log=args.vgt_log,
        )
        print(json.dumps({
            "som_batch_html_report": result["html_report"],
            "som_batch_json_report": result["json_report"],
            "folders": len(result["folders"]),
            "pages": result["pages"],
        }, ensure_ascii=False, indent=2))
        return

    if args.som:
        if args.instruction:
            result = run_visual_grounding_step(
                image_path=args.som,
                instruction=args.instruction,
                output_path=args.output,
                language=args.language,
                model=args.agent_model,
                log_path=args.vgt_log,
            )
            result["som_html_report"] = write_som_html_report(
                som_result=result["som"],
                agent_result=result["agent"],
                html_path=args.som_html,
                instruction=args.instruction,
                language=args.language,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        result = detect_ui_elements_set_of_mark(args.som, output_path=args.output)
        som_html_report = write_som_html_report(
            som_result=result,
            html_path=args.som_html,
            language=args.language,
        )
        append_vgt_baseline_record(args.vgt_log, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": str(args.som),
            "language": args.language,
            "marked_image_path": result["marked_image_path"],
            "som_html_report": som_html_report,
            "element_count": result["count"],
            "agent_result": None,
            "tap_result": None,
            "success": None,
        })
        result["som_html_report"] = som_html_report
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.batch or (args.baseline is None and args.actual is None):
        # Resolve pages dynamically based on baseline_folder contents and app config
        pages_to_compare = args.pages
        if args.flow or (pages_to_compare and "flow" in pages_to_compare):
            flow_pages = []
            if baseline_folder.exists():
                for f in sorted(baseline_folder.glob("*.png")):
                    stem = f.stem
                    if stem.startswith("."):
                        continue
                    if app_name and stem.startswith(f"{app_name}_"):
                        stem = stem[len(app_name)+1:]
                    if re.match(r"^\d{2}_", stem):
                        if stem not in flow_pages:
                            flow_pages.append(stem)
            pages_to_compare = flow_pages if flow_pages else args.pages
        elif pages_to_compare and "all" in pages_to_compare:
            if baseline_folder.exists():
                pages_to_compare = sorted([f.stem for f in baseline_folder.glob("*.png") if not f.name.startswith(".")])
        elif not pages_to_compare:
            cfg_page_names = []
            if cfg_flows and "steps" in cfg_flows[0]:
                for s in cfg_flows[0].get("steps", []):
                    pn = s.get("page_name")
                    if pn and pn not in cfg_page_names:
                        cfg_page_names.append(pn)

            matched = []
            if baseline_folder.exists():
                if cfg_page_names:
                    matched = [p for p in cfg_page_names if _find_screenshot_file(baseline_folder, p, app_name=app_name)]
                if not matched:
                    standard_pages = [
                        "home_page", "triage_location_menu", "scan_page", "overview_page",
                        "vehicle_page", "settings_page", "activation_token_page", "login_page", "language_dialog"
                    ]
                    matched = [p for p in standard_pages if _find_screenshot_file(baseline_folder, p, app_name=app_name)]
                if not matched:
                    # Flow step fallback (e.g. Bluesky or flow-based apps)
                    for f in sorted(baseline_folder.glob("*.png")):
                        stem = f.stem
                        if stem.startswith("."):
                            continue
                        if app_name and stem.startswith(f"{app_name}_"):
                            stem = stem[len(app_name)+1:]
                        if re.match(r"^\d{2}_", stem):
                            if stem not in matched:
                                matched.append(stem)
                if not matched:
                    matched = sorted([f.stem for f in baseline_folder.glob("*.png") if not f.name.startswith(".")])
            pages_to_compare = matched

        if args.version_baseline_root:
            previous_root = Path(args.version_baseline_root)
            current_root = Path(args.version_actual_root) if args.version_actual_root else base_dir
            folder_names = [baseline_folder.name] + [Path(f).name for f in actual_folders]
            folder_pairs = []
            for name in folder_names:
                if (previous_root / name).is_dir() and (current_root / name).is_dir():
                    folder_pairs.append((previous_root / name, current_root / name))
                else:
                    print(f"Skipping {name}: folder missing in {previous_root} or {current_root}")
            if not folder_pairs:
                parser.error(f"No matching language folders in {previous_root} and {current_root}")
            run_version_comparison(
                folder_pairs=folder_pairs,
                pages=pages_to_compare,
                output_root=output_root,
                baseline_root=previous_root,
                threshold=args.threshold,
                min_box_size=args.min_box_size,
                use_ai=use_ai,
                ai_model=ai_model,
                only_diffs=args.only_diffs,
                app=app_name,
                baseline_label=args.baseline_label,
                actual_label=args.actual_label,
                compare_options=compare_options,
            )
            return

        run_batch_comparison(
            baseline_folder=baseline_folder,
            actual_folders=actual_folders,
            pages=pages_to_compare,
            output_root=output_root,
            threshold=args.threshold,
            min_box_size=args.min_box_size,
            use_ai=use_ai,
            ai_model=ai_model,
            only_diffs=args.only_diffs,
            app=app_name,
            compare_options=compare_options,
        )
        return

    if args.baseline is None or args.actual is None:
        parser.error("Provide both baseline and actual paths, or use --batch")

    output_path = args.output or build_default_output_path(args.baseline, args.actual)
    pair_ignore_rects = _ignore_rects_for_page(compare_options.get("ignore_regions"), Path(args.actual).parent.name, Path(args.actual).stem)
    comparison_result = compare_ui_screenshots(
        args.baseline, args.actual, output_path, threshold=args.threshold, min_box_size=args.min_box_size,
        ignore_top=compare_options.get("mask_top", 140), ignore_bottom=compare_options.get("mask_bottom", 70),
        ignore_regions=pair_ignore_rects,
    )
    if args.ai_local:
        from common_agent.diff_triage import triage_page_local
        ai_analysis = triage_page_local(
            baseline_path=args.baseline,
            actual_path=args.actual,
            boxes=comparison_result["cv"]["boxes"],
            page_name=Path(args.actual).stem,
            model_path=ai_model,
            max_boxes=args.ai_max_boxes,
            output_path=Path(output_path).with_name(f"triage_{Path(output_path).stem}.png"),
        )
        print(json.dumps({"ai_analysis": ai_analysis}, ensure_ascii=False, indent=2))
    elif args.ai:
        ai_analysis = analyze_diff_with_ai(
            baseline_path=args.baseline,
            actual_path=args.actual,
            cv_diff_path=comparison_result["cv"]["output_path"],
            page_name=Path(args.actual).stem,
            cv_diff_count=comparison_result["cv"]["count"],
            model=args.ai_model,
        )
        print(json.dumps({"ai_analysis": ai_analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
