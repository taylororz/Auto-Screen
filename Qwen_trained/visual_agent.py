import os
import json
import time
import base64
import ssl
from pathlib import Path
from urllib import error, request
from datetime import datetime, timezone
import cv2
import numpy as np

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "Qwen3-VL-4B-V2")
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH") or DEFAULT_MODEL_DIR
_model = None
_processor = None


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


def _deduplicate_boxes(boxes):
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3])
    kept = []
    for box in sorted_boxes:
        if not any(_is_concentric_duplicate(box, k) or _iou(box, k) > 0.7 for k in kept):
            kept.append(box)
    return sorted(kept, key=lambda box: (box[1], box[0]))


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


def annotate_ui_elements(image_path, output_path=None, min_box_size=18, min_area=350, max_area_ratio=0.35, canny_low=50, canny_high=150, **kwargs):
    """
    CV detection: Identifies buttons and fields, saves Set-of-Mark image,
    and returns (elements_map, annotated_img_path, elements).
    - elements_map: {"1": (x, y), "2": (x, y)} (physical center coordinates)
    - annotated_img_path: path to the Set-of-Mark image
    - elements: detailed element list
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Edge-based mask
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_mask = cv2.dilate(edges, dilate_kernel, iterations=1)

    # 2. Threshold-based mask
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )
    thresh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threshold_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thresh_kernel)

    # Check for modal dialog
    dialog_rect = _detect_modal_dialog(image)
    boxes = []

    if dialog_rect is not None:
        dx, dy, dw, dh = dialog_rect
        roi_edge = edge_mask[dy:dy+dh, dx:dx+dw]
        roi_thresh = threshold_mask[dy:dy+dh, dx:dx+dw]

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        roi_t_mask = cv2.morphologyEx(roi_thresh.copy(), cv2.MORPH_CLOSE, close_kernel, iterations=1)
        roi_combined = cv2.bitwise_or(roi_edge, roi_t_mask)

        contours, _ = cv2.findContours(roi_combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        dialog_area = dw * dh
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if w < 16 or h < 14 or area < 250 or area > 0.40 * dialog_area:
                continue
            if (w / h) > 12 or (w / h) < 0.08:
                continue
            boxes.append((dx + x, dy + y, w, h))

        if boxes:
            boxes = _merge_adjacent_icon_bars(boxes, max_v_gap=20)
            boxes = _merge_horizontal_boxes(boxes, max_gap=40)
            boxes = _merge_vertical_text_lines(boxes, max_v_gap=16)
            boxes = [b for b in boxes if b[2] >= 18 and b[3] >= 16 and (b[2] * b[3]) >= 350]

            sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3])
            kept = []
            for b in sorted_boxes:
                if not any(_is_concentric_duplicate(b, k) or _iou(b, k) > 0.6 for k in kept):
                    kept.append(b)
            boxes = sorted(kept, key=lambda box: (box[1], box[0]))

    # Fallback to standard view if no modal dialog was detected or fewer than 2 elements were found
    if not boxes or len(boxes) < 2:
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
            # Ignore iOS status bar at the top
            if y < 110:
                continue
            if w < 14 or h < 8 or area < 250 or area > max_area:
                continue
            if aspect_ratio > 14 or aspect_ratio < 0.06:
                continue
            raw_boxes.append((x, y, w, h))

        boxes = _merge_adjacent_icon_bars(raw_boxes, max_v_gap=25)
        boxes = _merge_horizontal_boxes(boxes, max_gap=45)
        boxes = _merge_vertical_text_lines(boxes, max_v_gap=16)
        boxes = [b for b in boxes if b[2] >= 18 and b[3] >= 16 and (b[2] * b[3]) >= 400]

        sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        kept = []
        for box in sorted_boxes:
            def _intersection_over_area(box_a, box_b):
                ax, ay, aw, ah = box_a
                bx, by, bw, bh = box_b
                x_left = max(ax, bx)
                y_top = max(ay, by)
                x_right = min(ax + aw, bx + bw)
                y_bottom = min(ay + ah, by + bh)
                if x_right <= x_left or y_bottom <= y_top:
                    return 0
                return (x_right - x_left) * (y_bottom - y_top) / (aw * ah)

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

    # Save metadata as JSON
    json_path = output_path.with_suffix(".json")
    metadata = {
        "source_image": str(image_path),
        "annotated_image": str(output_path),
        "image_size": {"width": width, "height": height},
        "element_count": len(elements),
        "elements": elements,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return elements_map, str(output_path), elements


def _image_data_url(image_path):
    """Converts local image to Base64 Data URL."""
    image_file = Path(image_path)
    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    suffix = image_file.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime_type};base64,{encoded}"


def get_local_model_and_processor(model_path=None):
    """
    Lazily loads Qwen3-VL model and processor into memory (Singleton pattern).
    """
    global _model, _processor
    if _model is None or _processor is None:
        if model_path is None:
            model_path = MODEL_PATH
        print(f"🔄 Loading local fine-tuned Qwen3-VL model from: {model_path} ...")
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

        load_kwargs = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        )
        if not torch.cuda.is_available():
            _model.to(device)

        _processor = AutoProcessor.from_pretrained(model_path)
    return _model, _processor


def local_visual_agent_inference(image_path, instruction, model_path=None, valid_ids=None):
    """Uses local fine-tuned model for GUI element localization with image token optimization."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    model, processor = get_local_model_and_processor(model_path)

    if valid_ids:
        id_list_str = ", ".join(str(i) for i in sorted(valid_ids))
        prompt_text = (
            f"<image>This is a UI testing task with numbered Set-of-Mark bounding boxes. "
            f"Find the UI element that matches this instruction: '{instruction}'. "
            f"You MUST select one integer ID from the labeled elements on screen (valid IDs: {id_list_str}). "
            f"Output ONLY a valid integer ID in JSON format: {{\"target_id\": <selected_id>}}"
        )
    else:
        prompt_text = f"<image>This is a UI testing task. Find the element that matches this instruction: '{instruction}'. Output the target ID in JSON format."

    pil_img = Image.open(str(image_path))
    # LOCAL_MAX_IMAGE_EDGE: longest edge passed to the local model (default 1600, <= 0 keeps the original size)
    max_edge = int(os.environ.get("LOCAL_MAX_IMAGE_EDGE", "1600"))
    max_dim = max(pil_img.size)
    if max_edge > 0 and max_dim > max_edge:
        scale = float(max_edge) / max_dim
        new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    output_text = ""
    inputs = None
    generated_ids = None
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        device = next(model.parameters()).device

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=45, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    finally:
        del inputs, generated_ids
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_text


def _parse_ai_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def heuristic_fallback(instruction, elements):
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
    if any(keyword in instruction_lower for keyword in ["back", "back icon", "back arrow", "return", "go back", "drawer", "menu"]):
        return find_closest(physical_w * 0.08, physical_h * 0.096)

    # 2. Top Actions / Close / Cancel on top right
    if any(keyword in instruction_lower for keyword in ["close", "close button", "dismiss", "x", "cancel", "exit", "done"]):
        top_right_elems = [e for e in elements if e["center"]["x"] > physical_w * 0.6 and e["center"]["y"] < physical_h * 0.3]
        if top_right_elems:
            return min(top_right_elems, key=lambda e: (e["center"]["x"] - physical_w * 0.92)**2 + (e["center"]["y"] - physical_h * 0.096)**2)["id"]
        return find_closest(physical_w * 0.92, physical_h * 0.096)

    # 3. Bottom Navigation bar (tabs)
    bottom_y = physical_h - 150
    is_tab_query = any(k in instruction_lower for k in ["tab", "bottom bar", "tab bar", "nav bar", "navigation"])

    # 3.1 Tab 1 / Far left (~12-15% x): Home / Start / Main / Dashboard
    if any(k in instruction_lower for k in ["home", "start", "main", "dashboard"]) or (is_tab_query and "1" in instruction_lower):
        return find_closest(physical_w * 0.15, bottom_y)

    # 3.2 Tab 2 / Middle-left (~32-35% x): Search / Explore / Discover
    if any(k in instruction_lower for k in ["search", "explore", "discover", "find"]) or (is_tab_query and "2" in instruction_lower):
        return find_closest(physical_w * 0.35, bottom_y)

    # 3.3 Tab 3 / Center (~50% x): Central action / Feed / Chat / Message / Overview
    if any(k in instruction_lower for k in ["chat", "message", "feed", "create", "center", "overview", "inbox"]) or (is_tab_query and "3" in instruction_lower):
        return find_closest(physical_w * 0.50, bottom_y)

    # 3.4 Tab 4 / Middle-right (~68-70% x): Notifications / Activity / Bell / Alerts
    if any(k in instruction_lower for k in ["notification", "bell", "alerts", "activity", "status"]) or (is_tab_query and "4" in instruction_lower):
        return find_closest(physical_w * 0.70, bottom_y)

    # 3.5 Tab 5 / Far right (~88% x): Profile / User / Settings / Account
    if any(k in instruction_lower for k in ["profile", "account", "settings", "user", "me", "@"]) or (is_tab_query and "5" in instruction_lower):
        return find_closest(physical_w * 0.88, bottom_y)

    # 4. Form and interactive input elements
    # 4.1 Input text fields
    if any(k in instruction_lower for k in ["input", "text field", "field", "type", "search bar", "edit text"]):
        upper_inputs = [e for e in elements if e["center"]["y"] < physical_h * 0.5]
        if upper_inputs:
            return min(upper_inputs, key=lambda e: (e["center"]["x"] - physical_w * 0.5)**2 + (e["center"]["y"] - physical_h * 0.25)**2)["id"]
        return find_closest(physical_w * 0.5, physical_h * 0.25)

    # 4.2 Dropdown / Picker / Selection list
    if any(k in instruction_lower for k in ["dropdown", "select", "picker", "choice", "option", "menu"]):
        return find_closest(physical_w * 0.5, physical_h * 0.45)

    # 4.3 Confirmation / Submit / Save buttons
    if any(k in instruction_lower for k in ["submit", "continue", "save", "ok", "confirm", "login", "sign in", "register", "next", "send", "apply"]):
        action_buttons = [e for e in elements if e["center"]["y"] > physical_h * 0.4]
        if action_buttons:
            return min(action_buttons, key=lambda e: abs(e["center"]["x"] - physical_w * 0.5) + abs(e["center"]["y"] - physical_h * 0.85))["id"]
        return find_closest(physical_w * 0.5, physical_h * 0.85)

    # 5. Default: closest to center
    return find_closest(physical_w * 0.5, physical_h * 0.5)


def ask_ai_for_element_id(annotated_img_path, instruction, elements, language=None, model=None, timeout=60):
    """
    LLM decision: Calls local fine-tuned Qwen3-VL model to select target_id.
    """
    start_time = time.time()
    valid_ids = {element["id"] for element in elements}
    try:
        raw_output = local_visual_agent_inference(annotated_img_path, instruction, valid_ids=valid_ids)
        print(f"🤖 Local Qwen3-VL model output: {raw_output}")

        parsed = _parse_ai_json(raw_output)
        target_id = parsed.get("target_id")
        confidence = parsed.get("confidence", 1.0 if target_id is not None else 0.0)

        if target_id is not None and target_id not in valid_ids:
            if isinstance(target_id, str) and target_id.isdigit() and int(target_id) in valid_ids:
                target_id = int(target_id)
            else:
                target_id = None

        used_fallback = False
        if target_id is None:
            target_id = heuristic_fallback(instruction, elements)
            used_fallback = True
            print(f"⚠️ Local model output invalid or no match with element ID (raw: '{raw_output}'). Heuristic fallback triggered: target_id={target_id}")

        result = {
            "status": "analyzed_fallback" if used_fallback else "analyzed_local",
            "model": "local_qwen3_vl",
            "target_id": target_id,
            "confidence": confidence,
            "rationale": parsed.get("rationale") or raw_output,
            "candidate_ids": parsed.get("candidate_ids", []),
            "time_ms": (time.time() - start_time) * 1000,
        }
        print(f"🤖 Qwen3-VL: target_id={result['target_id']} confidence={result['confidence']}")
        return result
    except Exception as exc:
        print(f"⚠️ Local Qwen3-VL model inference exception: {exc}. Heuristic fallback triggered.")
        target_id = heuristic_fallback(instruction, elements)
        return {
            "status": "error",
            "reason": str(exc),
            "target_id": target_id,
            "confidence": 0.1,
            "rationale": f"Local model error fallback: {exc}",
            "time_ms": (time.time() - start_time) * 1000,
        }


class VisualNavigationError(Exception):
    """Custom exception raised when Visual Agent fails to resolve a target element."""
    pass


def log_interaction(language, step_name, instruction, action, target_id, center_x, center_y, screenshot_path, annotated_image_path, decision, tap_x=None, tap_y=None, run_id=None, loop_index=None, step_index=None):
    """
    Appends execution log record to diff_results/interaction_log.jsonl.
    """
    log_dir = Path(__file__).resolve().parent / "diff_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "interaction_log.jsonl"
    
    record = {
        "language": language,
        "step_name": step_name,
        "instruction": instruction,
        "action": action,
        "target_id": target_id if target_id is not None else -1,
        "center_x": center_x,
        "center_y": center_y,
        "screenshot_path": str(screenshot_path),
        "annotated_image_path": str(annotated_image_path),
        "decision": decision,
        "tap_x": tap_x,
        "tap_y": tap_y,
        "run_id": run_id,
        "loop_index": loop_index,
        "step_index": step_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_page_changed(img_path_a, img_path_b, threshold=0.005):
    """
    CV check: compares before and after screenshots to detect page layout changes.
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
