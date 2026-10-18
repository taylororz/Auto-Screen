import os
import re
import json
import time
import base64
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np

# ==============================================================================
# Model Path Configuration
# ==============================================================================
_BASE_DIR = Path(__file__).resolve().parent
_QWEN_V2_DIR = _BASE_DIR / "Qwen3-VL-4B-V2"
_KATAPP_DIR = _BASE_DIR / "Katapp-Qwen3-VL-Agent"

DEFAULT_MODEL_DIR = str(_QWEN_V2_DIR if _QWEN_V2_DIR.exists() else _KATAPP_DIR)
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH") or (
    DEFAULT_MODEL_DIR if os.path.exists(DEFAULT_MODEL_DIR) else str(_KATAPP_DIR)
)

_model = None
_processor = None
_LAST_PREDICTED_COORDINATES = {}  # Global mapping supporting generic_engine


def annotate_ui_elements(image_path, output_path=None, **kwargs):
    """
    Direct Grounding (DG) Mode: No OpenCV bounding box detection and no Set-of-Mark (SoM) tags!
    The original screenshot is used directly without drawing boxes or numerical IDs.

    Returns:
    - elements_map: Dynamic coordinate mapping (populated after model inference)
    - annotated_img_path: Path to the clean original screenshot (no SoM boxes)
    - elements: Empty list [] (no CV boxes extracted upfront in Direct Grounding)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    # In Direct Grounding mode, use the original image directly
    img_target_path = image_path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path != image_path:
            cv2.imwrite(str(output_path), image)
        img_target_path = output_path

    global _LAST_PREDICTED_COORDINATES
    _LAST_PREDICTED_COORDINATES = {}
    elements = []

    return _LAST_PREDICTED_COORDINATES, str(img_target_path), elements


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
        print(f"🔄 Loading local Qwen-VL Direct Grounding model from: {model_path} ...")
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_cls = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

        load_kwargs = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        _model = model_cls.from_pretrained(
            model_path,
            **load_kwargs
        )
        if not torch.cuda.is_available():
            _model.to(device)

        _processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28  # Optimize image tokens for faster inference
        )
        print("✅ Qwen-VL model & processor loaded successfully!")
    return _model, _processor


def _parse_direct_grounding_coordinates(output_text, image_width=1206, image_height=2622):
    """
    Extracts absolute click coordinates (x, y) from Qwen-VL model generation output.
    Supports multiple formats:
    1. JSON: {"point": [x, y]} or {"coordinates": [x, y]} or {"x": x, "y": y}
    2. JSON Bounding Box: {"bbox": [ymin, xmin, ymax, xmax]} -> calculates center
    3. Qwen-VL standard grounding token tags: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
    4. Coordinate tuples via Regex: (x, y) or [x, y]
    """
    target_x = None
    target_y = None
    confidence = 1.0
    rationale = output_text

    # 1. Attempt JSON parsing
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

        # Check point representations
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

    # 2. Qwen-VL standard grounding token tags: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
    if target_x is None or target_y is None:
        box_pattern = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', output_text)
        if box_pattern:
            y1, x1, y2, x2 = map(int, box_pattern.groups())
            norm_cx = (x1 + x2) / 2.0 / 1000.0
            norm_cy = (y1 + y2) / 2.0 / 1000.0
            target_x = norm_cx * image_width
            target_y = norm_cy * image_height

    # 3. Regex coordinate fallback: (x, y) or [x, y]
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

    # 4. Automatic coordinate normalization scaling [0, 1000] to image pixels
    if target_x is not None and target_y is not None:
        if 0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0:
            target_x = target_x * image_width
            target_y = target_y * image_height
        elif target_x <= 1000 and target_y <= 1000 and (image_height > 1200 or image_width > 1000):
            target_x = (target_x / 1000.0) * image_width
            target_y = (target_y / 1000.0) * image_height

        target_x = int(max(0, min(image_width, round(target_x))))
        target_y = int(max(0, min(image_height, round(target_y))))
        return target_x, target_y, confidence, rationale

    return None, None, 0.0, rationale


def local_visual_agent_inference(image_path, instruction, model_path=None):
    """
    Direct Grounding inference:
    Feeds clean original image directly into the model to predict click coordinates (x, y).
    """
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = get_local_model_and_processor(model_path)

    prompt_text = (
        f"<image>This is a UI testing task. Locate the UI element that matches this instruction: '{instruction}'. "
        f"Output the target interaction point in normalized coordinates [0, 1000] in JSON format: "
        f'{{"point": [x, y], "confidence": 1.0, "rationale": "..."}}'
    )

    from PIL import Image

    pil_img = Image.open(str(image_path))
    max_dim = max(pil_img.size)
    if max_dim > 1280:
        scale = 1280.0 / max_dim
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

    output_text = ""
    generated_ids = None
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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


def heuristic_fallback(instruction, elements=None, image_size=(1206, 2622)):
    """
    Universal heuristic fallback for mobile UI elements:
    Returns estimated physical coordinates (x, y) based on UI conventions if model inference fails.
    """
    if elements and isinstance(elements, (list, tuple)) and len(elements) == 2 and isinstance(elements[0], (int, float)):
        physical_w, physical_h = elements
    else:
        physical_w, physical_h = image_size

    instruction_lower = instruction.lower()

    # 1. Top Navigation / Header (Back button on top left)
    if any(k in instruction_lower for k in ["back", "back icon", "back arrow", "return", "go back", "top left corner"]):
        return int(physical_w * 0.08), int(physical_h * 0.096)

    # 2. Top Actions / Menu / Drawer / Close
    if any(k in instruction_lower for k in ["menu", "drawer", "hamburger"]):
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


def predict_target_coordinates(image_path, instruction, model_path=None):
    """
    Main function for Direct Grounding:
    Predicts click coordinates (target_x, target_y) directly from screenshot and instruction.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is not None:
        img_h, img_w = img.shape[:2]
    else:
        img_w, img_h = 1206, 2622

    start_time = time.time()
    try:
        raw_output = local_visual_agent_inference(image_path, instruction, model_path=model_path)
        print(f"🤖 [Direct Grounding] Qwen-VL model output: {raw_output}")

        target_x, target_y, confidence, rationale = _parse_direct_grounding_coordinates(
            raw_output, image_width=img_w, image_height=img_h
        )

        used_fallback = False
        if target_x is None or target_y is None:
            print(f"⚠️ [Direct Grounding] Coordinates not found in model output. Heuristic fallback activated.")
            target_x, target_y = heuristic_fallback(instruction, image_size=(img_w, img_h))
            used_fallback = True
            confidence = 0.5

        decision = {
            "status": "analyzed_fallback" if used_fallback else "analyzed_direct_grounding",
            "model": "qwen_vl_direct_grounding",
            "point": [target_x, target_y],
            "center_x": target_x,
            "center_y": target_y,
            "confidence": confidence,
            "rationale": rationale,
            "raw_output": raw_output,
            "time_ms": (time.time() - start_time) * 1000,
        }
        return (target_x, target_y), decision
    except Exception as exc:
        print(f"⚠️ [Direct Grounding] Inference exception: {exc}. Heuristic fallback activated.")
        target_x, target_y = heuristic_fallback(instruction, image_size=(img_w, img_h))
        decision = {
            "status": "error",
            "reason": str(exc),
            "point": [target_x, target_y],
            "center_x": target_x,
            "center_y": target_y,
            "confidence": 0.1,
            "rationale": f"Model error fallback: {exc}",
            "time_ms": (time.time() - start_time) * 1000,
        }
        return (target_x, target_y), decision


def ask_ai_for_element_id(annotated_img_path, instruction, elements=None, language=None, model=None, timeout=60):
    """
    Compatibility interface for generic_engine.py:
    Executes Direct Grounding with AI model, stores coordinates in _LAST_PREDICTED_COORDINATES
    and returns a structured decision dictionary.
    """
    global _LAST_PREDICTED_COORDINATES

    (target_x, target_y), decision = predict_target_coordinates(annotated_img_path, instruction, model_path=model)
    target_id = 1
    _LAST_PREDICTED_COORDINATES["1"] = (target_x, target_y)
    _LAST_PREDICTED_COORDINATES[str(target_id)] = (target_x, target_y)

    decision["target_id"] = target_id
    print(f"🤖 [Direct Grounding] Predicted coordinates: ({target_x}, {target_y}) (confidence={decision.get('confidence', 1.0)})")
    return decision


class VisualNavigationError(Exception):
    """Custom exception raised when Visual Agent fails to resolve a target element."""
    pass


def log_interaction(language, step_name, instruction, action, target_id, center_x, center_y, screenshot_path, annotated_image_path, decision, tap_x=None, tap_y=None):
    """
    Appends execution log record to diff_results/interaction_log_DG.jsonl.
    """
    log_dir = Path(__file__).resolve().parent / "diff_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_dg = log_dir / "interaction_log_DG.jsonl"

    record = {
        "language": language,
        "step_name": step_name,
        "instruction": instruction,
        "action": action,
        "mode": "direct_grounding",
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
    with log_file_dg.open("a", encoding="utf-8") as f:
        f.write(line)


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
