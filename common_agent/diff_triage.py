"""
Local VLM assessment of difference boxes (optional, not part of the evaluated system).

Each difference box of the image comparison is cut out of the baseline and the current
screenshot with a margin. Both crops are shown side by side to a local vision-language
model, which rates whether the current crop shows a display defect. The screenshots do not
leave the machine, so the assessment is compatible with the data protection requirement
that rules out the cloud assessment (compare_images.py --ai).

The verdicts are a suggestion that orders the boxes for the reviewer. They do not replace
the decision of a person.

Usage: compare_images.py --batch ... --ai-local [--ai-model <path or Hugging Face id>]
"""

import json
import os
import re
import time
from pathlib import Path

DEFAULT_TRIAGE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VERDICTS = ("defect", "acceptable", "unclear")
VERDICT_COLORS = {"defect": (0, 0, 230), "unclear": (0, 150, 255), "acceptable": (150, 150, 150), "not_rated": (255, 255, 255)}

PROMPT = (
    "You review a visual regression test of a mobile app. The image shows the same screen region twice: "
    "LEFT = reference build, RIGHT = new build. The region was flagged because its pixels differ. "
    "Decide whether the RIGHT side shows a display defect. Display defects are: text cut off or truncated, "
    "text overflowing or clipped by its container, overlapping elements, missing or garbled characters, "
    "missing elements, broken or misaligned layout, unreadable contrast. "
    "Different wording, a different language, changed dynamic content (time, counters, images) or a small "
    "shift without visual breakage are not defects. "
    'Answer with JSON only: {"verdict": "defect|acceptable|unclear", "category": "short category", '
    '"reason": "one short sentence"}'
)

_model = None
_processor = None
_loaded_model_path = None


def default_triage_model():
    """Model used for the triage: TRIAGE_MODEL_PATH or the Qwen3-VL-4B base model."""
    return os.environ.get("TRIAGE_MODEL_PATH") or DEFAULT_TRIAGE_MODEL


def _get_model_and_processor(model_path):
    """Lazily loads the triage model once per process (same loading as Qwen_trained/visual_agent_DG.py)."""
    global _model, _processor, _loaded_model_path
    if _model is not None and _loaded_model_path == model_path:
        return _model, _processor

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as model_cls
    except ImportError:
        from transformers import AutoModelForVision2Seq as model_cls

    print(f"🔄 Loading local triage model: {model_path} ...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    # Prefer the local copy (no request to the Hugging Face Hub). Download only if the model is not cached yet.
    try:
        _model = model_cls.from_pretrained(model_path, local_files_only=True, **load_kwargs)
        local_only = True
    except OSError:
        print("ℹ️ Triage model not found locally, downloading it once")
        _model = model_cls.from_pretrained(model_path, **load_kwargs)
        local_only = False
    if device != "cuda":
        _model.to(device)
    _processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    _loaded_model_path = model_path
    print("✅ Triage model loaded")
    return _model, _processor


def _crop_pair(base_img, actual_img, box, margin=24, max_edge=1280):
    """Cuts the box plus margin out of both screenshots and places the crops side by side (PIL image)."""
    from PIL import Image, ImageDraw

    x, y, w, h = box
    img_w, img_h = base_img.size
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(img_w, x + w + margin), min(img_h, y + h + margin)
    crop_base = base_img.crop((left, top, right, bottom))
    crop_actual = actual_img.crop((left, top, right, bottom))

    gap, label_h = 16, 34
    cw, ch = crop_base.size
    canvas = Image.new("RGB", (cw * 2 + gap, ch + label_h), (255, 255, 255))
    canvas.paste(crop_base, (0, label_h))
    canvas.paste(crop_actual, (cw + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((cw, 0, cw + gap, ch + label_h), fill=(40, 40, 40))
    draw.text((6, 8), "LEFT: reference", fill=(0, 0, 0))
    draw.text((cw + gap + 6, 8), "RIGHT: new build", fill=(0, 0, 0))

    scale = min(1.0, max_edge / max(canvas.size))
    if scale < 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.Resampling.LANCZOS)
    return canvas


def _parse_verdict(text):
    """Reads verdict, category and reason from the model answer (JSON or free text)."""
    parsed = {}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
        except Exception:
            parsed = {}
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        match = re.search(r"\b(defect|acceptable|unclear)\b", text.lower())
        verdict = match.group(1) if match else "unclear"
    return {
        "verdict": verdict,
        "category": str(parsed.get("category") or "").strip(),
        "reason": str(parsed.get("reason") or text).strip()[:300],
    }


def rate_box(base_img, actual_img, box, model_path, margin=24):
    """Rates one difference box with the local model."""
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _get_model_and_processor(model_path)
    pair_image = _crop_pair(base_img, actual_img, box, margin=margin)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pair_image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = None
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    finally:
        del inputs, generated_ids
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = _parse_verdict(output_text)
    result["raw_output"] = output_text
    return result


def _write_triage_image(actual_img, rated, unrated, output_path):
    """Draws the rated boxes on the current screenshot: red defect, orange unclear, gray acceptable, white not rated."""
    import cv2
    import numpy as np

    image = cv2.cvtColor(np.array(actual_img), cv2.COLOR_RGB2BGR)
    entries = [(r["box"], r["verdict"], r["index"]) for r in rated] + [(r["box"], "not_rated", r["index"]) for r in unrated]
    for (x, y, w, h), verdict, index in entries:
        color = VERDICT_COLORS[verdict]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
        cv2.putText(image, f"{index} {verdict}", (x, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def triage_page_local(baseline_path, actual_path, boxes, page_name=None, model_path=None, max_boxes=8, output_path=None, margin=24):
    """
    Rates the difference boxes of one page and returns a result in the schema of analyze_diff_with_ai()
    (severity, summary, likely_causes, recommended_actions, ignore_candidate), extended by the verdict per box.
    The largest boxes are rated first, at most max_boxes per page.
    """
    model_path = model_path or default_triage_model()
    base_result = {"backend": "local", "model": model_path, "likely_causes": [], "recommended_actions": [], "boxes": []}
    if not boxes:
        return {**base_result, "status": "skipped", "reason": "No CV differences detected", "severity": "pass",
                "summary": "No visual differences were detected by CV.", "ignore_candidate": True}

    from PIL import Image

    start_time = time.time()
    base_img = Image.open(str(baseline_path)).convert("RGB")
    actual_img = Image.open(str(actual_path)).convert("RGB")
    if actual_img.size != base_img.size:
        actual_img = actual_img.resize(base_img.size)

    indexed = [{"index": i, "box": [int(v) for v in box]} for i, box in enumerate(boxes, start=1)]
    by_area = sorted(indexed, key=lambda item: -(item["box"][2] * item["box"][3]))
    to_rate, unrated = by_area[:max_boxes], by_area[max_boxes:]

    rated = []
    try:
        for item in to_rate:
            verdict = rate_box(base_img, actual_img, item["box"], model_path, margin=margin)
            rated.append({**item, **verdict})
            print(f"   🧠 Box {item['index']} {item['box']}: {verdict['verdict']} ({verdict['category'] or '-'})")
    except Exception as exc:
        return {**base_result, "status": "error", "reason": f"Local triage failed: {exc}", "severity": "unknown",
                "summary": None, "ignore_candidate": None, "boxes": rated}

    defects = [r for r in rated if r["verdict"] == "defect"]
    unclear = [r for r in rated if r["verdict"] == "unclear"]
    if defects:
        severity = "fail"
    elif unclear or unrated:
        severity = "warn"
    else:
        severity = "pass"

    summary = f"Local VLM: {len(defects)} of {len(rated)} rated boxes look like a display defect, {len(unclear)} unclear"
    if unrated:
        summary += f", {len(unrated)} boxes not rated"
    result = {
        **base_result,
        "status": "analyzed",
        "severity": severity,
        "summary": summary + ".",
        "likely_causes": sorted({r["category"] for r in defects if r["category"]}),
        "recommended_actions": [f"Box {r['index']}: {r['reason']}" for r in sorted(defects + unclear, key=lambda r: r["index"])],
        "ignore_candidate": not defects and not unclear and not unrated,
        "boxes": sorted(rated + [{**item, "verdict": "not_rated"} for item in unrated], key=lambda r: r["index"]),
        "time_ms": (time.time() - start_time) * 1000,
    }
    if output_path:
        result["triage_image"] = _write_triage_image(actual_img, rated, unrated, output_path)
    print(f"   🤖 Local triage ({page_name or Path(actual_path).stem}): {severity} - {result['summary']}")
    return result
