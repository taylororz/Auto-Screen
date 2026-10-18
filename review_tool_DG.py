#!/usr/bin/env python3
import os
import re
import json
import math
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import gradio as gr

# ==============================================================================
# Dataset Configuration for Direct Grounding (DG)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_OPTIONS = {
    "Qwen_trained (Direct Grounding)": str(SCRIPT_DIR / "Qwen_trained/diff_results/interaction_log_DG.jsonl"),
    "Qwen_untrain (Direct Grounding)": str(SCRIPT_DIR / "Qwen_untrain/diff_results/interaction_log_DG.jsonl"),
    "claude (Direct Grounding)": str(SCRIPT_DIR / "claude/diff_results/interaction_log_DG.jsonl"),
    "openAI_6.0_Luna (DG)": str(SCRIPT_DIR / "openAI_6.0_Luna/diff_results/interaction_log_DG.jsonl"),
    "openAI_6.0_Sol (DG)": str(SCRIPT_DIR / "openAI_6.0_Sol/diff_results/interaction_log_DG.jsonl"),
    "openAI_5.6_Luna (DG)": str(SCRIPT_DIR / "openAI_5.6_Luna/diff_results/interaction_log_DG.jsonl"),
    "openAI_5.6_Terra (DG)": str(SCRIPT_DIR / "openAI_5.6_Terra/diff_results/interaction_log_DG.jsonl"),
    "iOS-1K Cross-App Eval (DG)": str(SCRIPT_DIR / "datasets/dataset_ios1k/ios1k_dg_eval.jsonl"),
    "iOS-1K Cross-App Train (DG)": str(SCRIPT_DIR / "datasets/dataset_ios1k/ios1k_dg_train.jsonl"),
}

def _find_default_log_file():
    for label, path in DATASET_OPTIONS.items():
        if os.path.exists(path):
            return path
    qwen_trained_path = str(SCRIPT_DIR / "Qwen_trained/diff_results/interaction_log.jsonl")
    if os.path.exists(qwen_trained_path):
        return qwen_trained_path
    return list(DATASET_OPTIONS.values())[0]

LOG_FILE = os.environ.get("REVIEW_LOG_FILE") or _find_default_log_file()
CLEAN_FILE = os.environ.get("REVIEW_CLEAN_FILE") or os.path.join(os.path.dirname(LOG_FILE), "finetune_dataset_DG.jsonl")

logs = []
TOTAL_LOGS = 0
current_idx = 0


def get_reviewed_timestamps():
    """Reads all timestamps already saved in finetune_dataset_DG.jsonl."""
    if not os.path.exists(CLEAN_FILE):
        return set()
    reviewed = set()
    with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "timestamp" in data:
                        reviewed.add(data["timestamp"])
                except Exception:
                    pass
    return reviewed


def get_unreviewed_logs():
    """Returns pure Direct Grounding interaction logs."""
    if not os.path.exists(LOG_FILE):
        return [], 0
    reviewed_ts = get_reviewed_timestamps()
    total_count = 0
    unreviewed = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Strict filter: exclude old SoM box logs
                    if data.get("mode") != "direct_grounding" and "_som.png" in str(data.get("annotated_image_path", "")):
                        continue
                    if data.get("center_x") is None or data.get("center_y") is None:
                        continue
                    
                    total_count += 1
                    if data.get("timestamp") not in reviewed_ts:
                        unreviewed.append(data)
                except Exception:
                    pass
    return unreviewed, total_count


def resolve_image_path(img_path):
    """Dynamically resolves image path."""
    if not img_path:
        return None
    if os.path.exists(img_path):
        return img_path

    filename = os.path.basename(img_path)
    log_dir = os.path.dirname(os.path.abspath(LOG_FILE))

    for root, _, files in os.walk(log_dir):
        if filename in files:
            return os.path.join(root, filename)

    parts = img_path.split("states/")
    if len(parts) > 1:
        rel_state = os.path.join("states", parts[-1])
        c1 = os.path.join(log_dir, rel_state)
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(os.path.dirname(log_dir), rel_state)
        if os.path.exists(c2):
            return c2

    return img_path


def render_direct_grounding_overlay(image_path, ai_point=None, gt_point=None):
    """
    Renders high-precision reticle / bullseye on the screenshot.
    - AI Prediction: Red reticle with label (AI: x, y)
    - Ground Truth: Green reticle with label (GT: x, y)
    """
    if not image_path or not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    overlay = img.copy()

    # 1. AI Predicted Point (Red)
    if ai_point and ai_point[0] is not None and ai_point[1] is not None:
        ax, ay = int(ai_point[0]), int(ai_point[1])
        ax = max(0, min(w - 1, ax))
        ay = max(0, min(h - 1, ay))

        cv2.circle(overlay, (ax, ay), 26, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (ax, ay), 12, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (ax, ay), 4, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.line(overlay, (ax - 36, ay), (ax + 36, ay), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, (ax, ay - 36), (ax, ay + 36), (0, 0, 255), 2, cv2.LINE_AA)

        label = f"AI Pred: ({ax}, {ay})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = min(w - tw - 12, max(10, ax + 15))
        ty = max(30, ay - 15)
        cv2.rectangle(overlay, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 6), (0, 0, 0), -1)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

    # 2. Ground Truth Point (Green)
    if gt_point and gt_point[0] is not None and gt_point[1] is not None:
        gx, gy = int(gt_point[0]), int(gt_point[1])
        gx = max(0, min(w - 1, gx))
        gy = max(0, min(h - 1, gy))

        cv2.circle(overlay, (gx, gy), 26, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(overlay, (gx, gy), 12, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(overlay, (gx, gy), 4, (0, 255, 0), -1, cv2.LINE_AA)

        cv2.line(overlay, (gx - 36, gy), (gx + 36, gy), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(overlay, (gx, gy - 36), (gx, gy + 36), (0, 255, 0), 2, cv2.LINE_AA)

        label = f"GT: ({gx}, {gy})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = min(w - tw - 12, max(10, gx + 15))
        ty = min(h - 15, gy + 35)
        cv2.rectangle(overlay, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 6), (0, 0, 0), -1)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        # Connection line (error vector)
        if ai_point and ai_point[0] is not None and ai_point[1] is not None:
            cv2.line(overlay, (ax, ay), (gx, gy), (255, 255, 0), 2, cv2.LINE_AA)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def calculate_statistics():
    """Calculates metrics & distance error for Direct Grounding including is_action_successful."""
    reviewed_records = []
    if os.path.exists(CLEAN_FILE):
        with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        reviewed_records.append(json.loads(line))
                    except Exception:
                        pass

    total_reviewed = len(reviewed_records)
    total_raw_logs = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            total_raw_logs = sum(1 for line in f if line.strip())

    if total_reviewed == 0:
        return f"""### 📊 Direct Grounding Error Rates & Metrics (Dataset: `{LOG_FILE}`)
*Currently there are **{total_raw_logs}** execution logs in total. No reviewed DG data available yet. Please mark and save ground truth points to generate statistics.*
"""

    correct_count = 0
    error_count = 0
    action_success_count = 0
    discarded_count = 0
    distances = []
    lang_stats = {}
    step_stats = {}

    for r in reviewed_records:
        lang = r.get("language", "Unknown")
        step = r.get("step_name", "Unknown")

        if lang not in lang_stats:
            lang_stats[lang] = {"total": 0, "correct": 0, "error": 0, "action_success": 0, "dists": []}
        if step not in step_stats:
            step_stats[step] = {"total": 0, "correct": 0, "error": 0, "action_success": 0, "dists": []}

        if r.get("discarded", False):
            discarded_count += 1
            continue

        ai_x = r.get("center_x")
        ai_y = r.get("center_y")
        gt_x = r.get("gt_x")
        gt_y = r.get("gt_y")
        is_correct = r.get("is_correct", False)
        is_action_succ = r.get("is_action_successful", is_correct)

        dist = None
        if ai_x is not None and ai_y is not None and gt_x is not None and gt_y is not None:
            dist = math.sqrt((ai_x - gt_x)**2 + (ai_y - gt_y)**2)
            distances.append(dist)
            lang_stats[lang]["dists"].append(dist)
            step_stats[step]["dists"].append(dist)

        lang_stats[lang]["total"] += 1
        step_stats[step]["total"] += 1

        if is_correct:
            correct_count += 1
            lang_stats[lang]["correct"] += 1
            step_stats[step]["correct"] += 1
        else:
            error_count += 1
            lang_stats[lang]["error"] += 1
            step_stats[step]["error"] += 1

        if is_action_succ:
            action_success_count += 1
            lang_stats[lang]["action_success"] += 1
            step_stats[step]["action_success"] += 1

    valid_reviewed = total_reviewed - discarded_count
    acc_rate = (correct_count / valid_reviewed * 100) if valid_reviewed > 0 else 0.0
    err_rate = (error_count / valid_reviewed * 100) if valid_reviewed > 0 else 0.0
    action_succ_rate = (action_success_count / valid_reviewed * 100) if valid_reviewed > 0 else 0.0
    avg_dist = (sum(distances) / len(distances)) if distances else 0.0

    report_md = f"""### 🎯 Direct Grounding (DG) Evaluation Metrics & Distance Analysis
> **Current Log File**: `{LOG_FILE}`  
> **DG Ground Truth File**: `{CLEAN_FILE}`

| Metric | Value / Count | Percentage |
| :--- | :--- | :--- |
| **Total Execution Logs** | {total_raw_logs} entries | 100% |
| **Reviewed Samples** | {total_reviewed} entries | {(total_reviewed/total_raw_logs*100 if total_raw_logs else 0):.1f}% |
| **Valid Evaluation Samples** | {valid_reviewed} entries | {(valid_reviewed/total_reviewed*100 if total_reviewed else 0):.1f}% |
| **⚡ Interaction Action Successful** | {action_success_count} entries | **{action_succ_rate:.2f}%** |
| **✅ Localization Accurate** | {correct_count} entries | **{acc_rate:.2f}%** |
| **❌ Localization Inaccurate / Corrected** | {error_count} entries | **{err_rate:.2f}%** |
| **📏 Mean Pixel Error Distance** | **{avg_dist:.2f} px** | - |
| **🗑️ Discarded Samples** | {discarded_count} entries | - |

#### 🌐 Direct Grounding Accuracy by Language

| Language | Samples | Action Successful | Localization Accurate | **Action Success Rate** | **Ø Distance Error** |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for lang, s in lang_stats.items():
        if s["total"] > 0:
            l_act = s["action_success"] / s["total"] * 100
            l_avg_d = (sum(s["dists"]) / len(s["dists"])) if s["dists"] else 0.0
            report_md += f"| {lang} | {s['total']} | {s['action_success']} | {s['correct']} | **{l_act:.2f}%** | {l_avg_d:.1f} px |\n"

    report_md += """
#### 📋 Direct Grounding Accuracy by Interaction Step

| Step Name | Samples | Action Successful | Localization Accurate | **Action Success Rate** | **Ø Distance Error** |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for step, s in step_stats.items():
        if s["total"] > 0:
            st_act = s["action_success"] / s["total"] * 100
            st_avg_d = (sum(s["dists"]) / len(s["dists"])) if s["dists"] else 0.0
            report_md += f"| `{step}` | {s['total']} | {s['action_success']} | {s['correct']} | **{st_act:.2f}%** | {st_avg_d:.1f} px |\n"

    return report_md


def switch_dataset(selected_option):
    global LOG_FILE, CLEAN_FILE, current_idx
    if selected_option in DATASET_OPTIONS:
        LOG_FILE = DATASET_OPTIONS[selected_option]
    else:
        LOG_FILE = selected_option
    CLEAN_FILE = os.path.join(os.path.dirname(LOG_FILE), "finetune_dataset_DG.jsonl")
    current_idx = 0

    rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(0)
    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def _save_record_to_clean_file(record):
    """Saves a record idempotently to CLEAN_FILE (replaces matching timestamp, otherwise appends)."""
    os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
    target_ts = record.get("timestamp")
    existing_lines = []
    replaced = False
    if os.path.exists(CLEAN_FILE):
        with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if target_ts and d.get("timestamp") == target_ts:
                            existing_lines.append(json.dumps(record, ensure_ascii=False) + '\n')
                            replaced = True
                        else:
                            existing_lines.append(line.rstrip('\n') + '\n')
                    except Exception:
                        existing_lines.append(line)
    if not replaced:
        existing_lines.append(json.dumps(record, ensure_ascii=False) + '\n')
    with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
        f.writelines(existing_lines)


def load_record(idx=0, target_record=None):
    global logs, TOTAL_LOGS, current_idx
    logs, TOTAL_LOGS = get_unreviewed_logs()
    reviewed_count = TOTAL_LOGS - len(logs)

    if target_record is not None:
        record = target_record
        found_idx = -1
        for i, item in enumerate(logs):
            if item.get("timestamp") == record.get("timestamp"):
                found_idx = i
                break
        idx_display = found_idx + 1 if found_idx >= 0 else len(logs)
        if found_idx >= 0:
            current_idx = found_idx
    else:
        if not logs or idx >= len(logs):
            progress_info = f"🎉 All entries in the current DG dataset have been reviewed! (Progress: {reviewed_count}/{TOTAL_LOGS})"
            return None, progress_info, "", "", "", True, "{}"
        current_idx = max(0, min(idx, len(logs) - 1))
        record = logs[current_idx]
        idx_display = current_idx + 1

    img_path = resolve_image_path(record.get("screenshot_path") or record.get("annotated_image_path", ""))

    ai_x = record.get("center_x")
    ai_y = record.get("center_y")
    gt_x = record.get("gt_x") if record.get("gt_x") is not None else ai_x
    gt_y = record.get("gt_y") if record.get("gt_y") is not None else ai_y
    action_succ = record.get("is_action_successful", True)

    rendered_img = render_direct_grounding_overlay(img_path, ai_point=(ai_x, ai_y), gt_point=(gt_x, gt_y))

    progress_info = f"### 📊 Review Progress: Pending {idx_display}/{len(logs)} | Completed: {reviewed_count}/{TOTAL_LOGS}\n"
    instruction = f"{progress_info}### 📋 Instruction: `{record.get('instruction', '')}`\n**Interaction Step**: `{record.get('step_name')}` | **App Locale**: `{record.get('language')}`"

    confidence = record.get("decision", {}).get("confidence", 1.0)
    rationale = record.get("decision", {}).get("rationale", "")

    ai_info = f"🎯 **Predicted Coordinates**: `({ai_x}, {ai_y})` (Confidence: {confidence:.2f})\n"
    if rationale:
        ai_info += f"💡 *Model Rationale*: {rationale}"

    current_rec_json = json.dumps(record, ensure_ascii=False)
    return rendered_img, instruction, ai_info, str(gt_x if gt_x is not None else ""), str(gt_y if gt_y is not None else ""), action_succ, current_rec_json


def on_image_click(evt: gr.SelectData, rec_json):
    """Allows setting the ground truth coordinate by clicking directly on the screenshot."""
    if not rec_json or rec_json == "{}":
        return None, "", ""
    try:
        record = json.loads(rec_json)
    except Exception:
        return None, "", ""

    img_path = resolve_image_path(record.get("screenshot_path") or record.get("annotated_image_path", ""))
    if not img_path or not os.path.exists(img_path):
        return None, "", ""

    img = cv2.imread(img_path)
    if img is None:
        return None, "", ""

    click_x = int(evt.index[0])
    click_y = int(evt.index[1])

    ai_x = record.get("center_x")
    ai_y = record.get("center_y")

    rendered_img = render_direct_grounding_overlay(img_path, ai_point=(ai_x, ai_y), gt_point=(click_x, click_y))
    return rendered_img, str(click_x), str(click_y)


def save_as_correct(rec_json, is_action_successful=True):
    """Marks AI prediction as 100% accurate (GT = AI Pred)."""
    global current_idx, logs
    if not rec_json or rec_json == "{}":
        rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
        return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, calculate_statistics()

    record = json.loads(rec_json)
    record["is_correct"] = True
    record["is_action_successful"] = bool(is_action_successful)
    record["gt_x"] = record.get("center_x")
    record["gt_y"] = record.get("center_y")
    record["discarded"] = False

    _save_record_to_clean_file(record)

    logs, _ = get_unreviewed_logs()
    if current_idx >= len(logs):
        current_idx = max(0, len(logs) - 1)
    rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def save_corrected(rec_json, manual_x, manual_y, is_action_successful=True):
    """Saves human-calibrated coordinates as Ground Truth."""
    global current_idx, logs
    if not rec_json or rec_json == "{}":
        rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
        return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, calculate_statistics()

    record = json.loads(rec_json)
    try:
        gt_x = int(manual_x.strip())
        gt_y = int(manual_y.strip())
    except (ValueError, AttributeError):
        gt_x = record.get("center_x")
        gt_y = record.get("center_y")

    ai_x = record.get("center_x")
    ai_y = record.get("center_y")

    dist = math.sqrt((ai_x - gt_x)**2 + (ai_y - gt_y)**2) if (ai_x and ai_y and gt_x and gt_y) else 0.0
    record["is_correct"] = (dist <= 40.0)
    record["is_action_successful"] = bool(is_action_successful)
    record["gt_x"] = gt_x
    record["gt_y"] = gt_y
    record["pixel_error_distance"] = dist
    record["discarded"] = False

    _save_record_to_clean_file(record)

    logs, _ = get_unreviewed_logs()
    if current_idx >= len(logs):
        current_idx = max(0, len(logs) - 1)
    rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def discard_sample(rec_json):
    """Discards current click sample."""
    global current_idx, logs
    if not rec_json or rec_json == "{}":
        rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
        return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, calculate_statistics()

    record = json.loads(rec_json)
    record["discarded"] = True
    record["is_correct"] = False
    record["is_action_successful"] = False

    _save_record_to_clean_file(record)

    logs, _ = get_unreviewed_logs()
    if current_idx >= len(logs):
        current_idx = max(0, len(logs) - 1)
    rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def undo_last():
    """Reverts the last decision in finetune_dataset_DG.jsonl and reloads that record for editing."""
    global current_idx, logs
    last_record = None
    if os.path.exists(CLEAN_FILE):
        with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
        if lines:
            last_line = lines[-1].strip()
            try:
                last_record = json.loads(last_line)
            except Exception:
                pass
            with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
                f.writelines((l if l.endswith('\n') else l + '\n') for l in lines[:-1])
            step_desc = last_record.get('step_name', '') if last_record else ''
            print(f"⬅️ Reverted last decision in {CLEAN_FILE} ({step_desc})")

    if last_record is not None:
        rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(target_record=last_record)
    else:
        rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(0)

    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def skip_sample():
    """Skips current sample and loads the next unreviewed sample."""
    global current_idx, logs
    logs, _ = get_unreviewed_logs()
    if logs:
        current_idx = (current_idx + 1) % len(logs)
    rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json = load_record(current_idx)
    stats_md = calculate_statistics()
    return rendered_img, instruction, ai_info, default_x, default_y, action_succ, current_rec_json, stats_md


def export_qwen_vl_dg_sft_dataset():
    """
    Converts reviewed Direct Grounding records into Qwen-VL SFT training format:
    Normalized coordinates [0, 1000] -> {"point": [x_norm, y_norm]}
    """
    if not os.path.exists(CLEAN_FILE):
        return "❌ No reviewed data found in finetune_dataset_DG.jsonl."

    output_path = Path(CLEAN_FILE).parent / "qwen_vl_dg_sft_dataset.json"
    sft_data = []

    with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            if r.get("discarded", False):
                continue

            img_path = resolve_image_path(r.get("screenshot_path") or r.get("annotated_image_path", ""))
            if not img_path or not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_x = r.get("gt_x", r.get("center_x"))
            gt_y = r.get("gt_y", r.get("center_y"))
            if gt_x is None or gt_y is None:
                continue

            norm_x = int(round((gt_x / w) * 1000))
            norm_y = int(round((gt_y / h) * 1000))

            instruction = r.get("instruction", "")

            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"<image>This is a UI testing task. Locate the UI element that matches this instruction: '{instruction}'. "
                            f"Output the target interaction point in normalized coordinates [0, 1000] in JSON format: "
                            f'{{"point": [x, y], "confidence": 1.0, "rationale": "..."}}'
                        )
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"point": [norm_x, norm_y], "confidence": 1.0}, ensure_ascii=False)
                    }
                ],
                "images": [
                    img_path
                ]
            }
            sft_data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(sft_data, out_f, ensure_ascii=False, indent=2)

    return f"✅ Successfully exported {len(sft_data)} SFT records to:\n`{output_path}`"


# ==============================================================================
# Gradio UI Layout
# ==============================================================================
initial_label = list(DATASET_OPTIONS.keys())[0]
for k, v in DATASET_OPTIONS.items():
    if v == LOG_FILE:
        initial_label = k
        break

def add_custom_dg_element(raw_json_state, custom_instruction, custom_x, custom_y, action_success):
    try:
        record = json.loads(raw_json_state)
    except Exception:
        record = {}

    if not custom_instruction or not custom_instruction.strip():
        return "⚠️ **Error**: Please enter a valid task instruction!", "", "", calculate_statistics()

    try:
        cx = int(custom_x) if custom_x and str(custom_x).strip() else record.get("center_x", 0)
        cy = int(custom_y) if custom_y and str(custom_y).strip() else record.get("center_y", 0)
    except ValueError:
        cx, cy = 0, 0

    img_path = record.get("screenshot_path", "")
    img_w, img_h = 1206, 2622
    if img_path and os.path.exists(img_path):
        im = cv2.imread(img_path)
        if im is not None:
            img_h, img_w = im.shape[:2]

    norm_x = int(round((cx / img_w) * 1000.0))
    norm_y = int(round((cy / img_h) * 1000.0))

    new_record = {
        "language": record.get("language", "English"),
        "step_name": f"{record.get('step_name', 'custom')}_manual_add",
        "instruction": custom_instruction.strip(),
        "action": "click",
        "mode": "direct_grounding",
        "target_id": 1,
        "center_x": cx,
        "center_y": cy,
        "screenshot_path": img_path,
        "annotated_image_path": img_path,
        "decision": {
            "status": "ground_truth_manually_added",
            "model": "human_expert",
            "point": [norm_x, norm_y],
            "center_x": cx,
            "center_y": cy,
            "confidence": 1.0,
            "rationale": f"Manually added ground truth for '{custom_instruction.strip()}'.",
            "raw_output": json.dumps({"point": [norm_x, norm_y], "confidence": 1.0})
        },
        "tap_x": cx,
        "tap_y": cy,
        "timestamp": f"dg_manual_add_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "is_correct": True,
        "is_action_successful": bool(action_success),
        "gt_x": cx,
        "gt_y": cy,
        "discarded": False,
        "dataset_source": record.get("dataset_source", "manual_addition")
    }

    os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
    with open(CLEAN_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(new_record, ensure_ascii=False) + '\n')

    status_msg = f"🎉 **Successfully added & saved as DG Ground Truth!**\n- **Instruction**: `{custom_instruction.strip()}`\n- **Coordinates**: `({cx}, {cy})` [Norm: `{norm_x}, {norm_y}`]"
    return status_msg, "", "", calculate_statistics()


with gr.Blocks(title="🎯 Auto-Screen Direct Grounding Review Tool") as demo:
    gr.Markdown("# 🎯 Auto-Screen Direct Grounding (DG) - Human Review & Evaluation Tool")
    gr.Markdown("*Visual evaluation of visual coordinate regression without SoM bounding boxes | Click accuracy & distance analysis*")

    current_rec_state = gr.State(value="{}")

    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=list(DATASET_OPTIONS.keys()),
            value=initial_label,
            label="📁 Select Direct Grounding Model Dataset",
            interactive=True,
            allow_custom_value=True
        )

    with gr.Tab("🔍 Direct Grounding Review & Annotation"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("💡 **Click directly on the image** to place the Ground Truth reticle (GT 🟢)!")
                image_display = gr.Image(type="numpy", interactive=True, label="Screenshot with AI Prediction 🎯 (Red) & GT 🟢 (Green)")

            with gr.Column(scale=2):
                instruction_text = gr.Markdown()
                ai_info_text = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### 📍 Ground Truth Coordinates & Action Outcome")
                with gr.Row():
                    gt_x_input = gr.Textbox(label="Ground Truth X (Pixel)")
                    gt_y_input = gr.Textbox(label="Ground Truth Y (Pixel)")

                action_success_checkbox = gr.Checkbox(
                    label="⚡ Interaction action executed successfully (is_action_successful)",
                    value=True,
                    interactive=True
                )

                with gr.Row():
                    correct_btn = gr.Button("✅ Accurate Prediction (AI Click Correct)", variant="primary")
                    save_custom_btn = gr.Button("📍 Save Calibrated Coordinates (Save GT)", variant="secondary")

                with gr.Row():
                    discard_btn = gr.Button("🗑️ Discard Sample", variant="stop")
                    prev_btn = gr.Button("⬅️ Undo Last Decision", variant="secondary")
                    skip_btn = gr.Button("⏭️ Skip Sample", variant="secondary")

                gr.Markdown("---")
                with gr.Accordion("➕ Manually Add Missing Element / New Instruction", open=True):
                    gr.Markdown("If an element was overlooked on this screen, click on the image on the left and enter the instruction:")
                    add_dg_instruction = gr.Textbox(label="📋 New Instruction", placeholder="e.g. Click the back button / Click the close button")
                    with gr.Row():
                        add_dg_x = gr.Textbox(label="📍 X-Coordinate (Pixel)", placeholder="Click on image or enter manually")
                        add_dg_y = gr.Textbox(label="📍 Y-Coordinate (Pixel)", placeholder="Click on image or enter manually")
                    add_dg_btn = gr.Button("➕ Save as New Ground Truth Click Point", variant="primary")
                    add_dg_status = gr.Markdown()

    with gr.Tab("📊 Real-time Error Rates & Distance Metrics (Dashboard)"):
        stats_display = gr.Markdown(value=calculate_statistics())
        with gr.Row():
            refresh_stats_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
            export_sft_btn = gr.Button("🚀 Export Qwen-VL SFT Training Dataset", variant="primary")
        export_result_text = gr.Markdown()

    # Event Handlers
    dataset_dropdown.change(
        fn=switch_dataset,
        inputs=[dataset_dropdown],
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    image_display.select(
        fn=on_image_click,
        inputs=[current_rec_state],
        outputs=[image_display, gt_x_input, gt_y_input]
    ).then(
        fn=lambda x, y: (x, y),
        inputs=[gt_x_input, gt_y_input],
        outputs=[add_dg_x, add_dg_y]
    )

    correct_btn.click(
        fn=save_as_correct,
        inputs=[current_rec_state, action_success_checkbox],
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    save_custom_btn.click(
        fn=save_corrected,
        inputs=[current_rec_state, gt_x_input, gt_y_input, action_success_checkbox],
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    discard_btn.click(
        fn=discard_sample,
        inputs=[current_rec_state],
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    prev_btn.click(
        fn=undo_last,
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    skip_btn.click(
        fn=skip_sample,
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )

    add_dg_btn.click(
        fn=add_custom_dg_element,
        inputs=[current_rec_state, add_dg_instruction, add_dg_x, add_dg_y, action_success_checkbox],
        outputs=[add_dg_status, add_dg_instruction, add_dg_x, stats_display]
    )

    refresh_stats_btn.click(
        fn=calculate_statistics,
        outputs=[stats_display]
    )

    export_sft_btn.click(
        fn=export_qwen_vl_dg_sft_dataset,
        outputs=[export_result_text]
    )

    def _init_page():
        r = load_record(current_idx)
        return r[:5] + (r[5], r[6], calculate_statistics())

    demo.load(
        fn=_init_page,
        inputs=[],
        outputs=[image_display, instruction_text, ai_info_text, gt_x_input, gt_y_input, action_success_checkbox, current_rec_state, stats_display]
    )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7861))
    print(f"🚀 Starting Auto-Screen Direct Grounding Review Tool on port: {port} ...")
    home_dir = str(Path.home())
    allowed = [home_dir, "/Users/taylorzhuang", "/Users/taylorzhuang/social-app", "/Users/taylorzhuang/Masterarbeit"]
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=False, allowed_paths=allowed)
    except OSError:
        demo.launch(server_name="127.0.0.1", share=False, allowed_paths=allowed)
