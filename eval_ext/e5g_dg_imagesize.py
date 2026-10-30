#!/usr/bin/env python3
"""E5g: Check image resolution (1280 px vs. original) used in base-model Direct Grounding runs."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG = ROOT_DIR / "Qwen_untrain" / "diff_results" / "interaction_log_DG.jsonl"


def run(client, image, instruction, resize):
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    model, processor = client._get_model_and_processor()
    prompt_text = (
        f"<image>This is a UI testing task. Locate the UI element that matches this instruction: '{instruction}'. "
        f"Output the target interaction point in normalized coordinates [0, 1000] in JSON format: "
        f'{{"point": [x, y], "confidence": 1.0, "rationale": "..."}}'
    )
    img = Image.open(image)
    if resize and max(img.size) > 1280:
        scale = 1280.0 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    return processor.batch_decode([ids[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]


def point_of(text):
    m = re.search(r'"point"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return [int(m.group(1)), int(m.group(2))] if m else None


def main(a):
    sys.path.insert(0, str(ROOT_DIR))
    from common_agent.model_client import QwenUntrainClient
    client = QwenUntrainClient()
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()][: a.steps]
    out = []
    for r in rows:
        logged = r["decision"].get("rationale", "")
        res = {"step": r["step_name"], "logged_point": point_of(logged), "logged_head": logged[:120]}
        for name, resize in (("a_1280", True), ("b_ohne", False)):
            raw = run(client, r["screenshot_path"], r["instruction"], resize)
            res[name] = {"point": point_of(raw), "same_point": point_of(raw) == point_of(logged),
                         "same_head": raw[:120] == logged[:120], "head": raw[:120]}
        out.append(res)
        print(json.dumps({k: (v if k != "logged_head" else None) for k, v in res.items()}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
