import os
import json
from pathlib import Path
import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_OPTIONS = {
    "Qwen_trained (Fine-tuned Model)": str(SCRIPT_DIR / "Qwen_trained/diff_results/interaction_log.jsonl"),
    "Qwen_untrain (Base Model)": str(SCRIPT_DIR / "Qwen_untrain/diff_results/interaction_log.jsonl"),
    "claude (Claude Sonnet 5.5 Model)": str(SCRIPT_DIR / "claude/diff_results/interaction_log.jsonl"),
    "openAI_6.0_Luna (GPT-6 Luna Model)": str(SCRIPT_DIR / "openAI_6.0_Luna/diff_results/interaction_log.jsonl"),
    "openAI_6.0_Sol (GPT-6 Sol Model)": str(SCRIPT_DIR / "openAI_6.0_Sol/diff_results/interaction_log.jsonl"),
    "openAI_5.6_Luna (GPT-4o Luna Model)": str(SCRIPT_DIR / "openAI_5.6_Luna/diff_results/interaction_log.jsonl"),
    "openAI_5.6_Terra (GPT-4o Terra Model)": str(SCRIPT_DIR / "openAI_5.6_Terra/diff_results/interaction_log.jsonl"),
    "Auto-Screen Root Log": str(SCRIPT_DIR / "diff_results/interaction_log.jsonl"),
    "iOS-1K SoM Eval": str(SCRIPT_DIR / "datasets/dataset_ios1k/ios1k_som_eval.jsonl"),
    "iOS-1K SoM Train": str(SCRIPT_DIR / "datasets/dataset_ios1k/ios1k_som_train.jsonl"),
}

def _find_default_log_file():
    for label, path in DATASET_OPTIONS.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    for label, path in DATASET_OPTIONS.items():
        if os.path.exists(path):
            return path
    return list(DATASET_OPTIONS.values())[0]

LOG_FILE = os.environ.get("REVIEW_LOG_FILE") or _find_default_log_file()
CLEAN_FILE = os.environ.get("REVIEW_CLEAN_FILE") or os.path.join(os.path.dirname(LOG_FILE), "finetune_dataset.jsonl")


def get_reviewed_timestamps():
    """Reads all timestamps already saved in finetune_dataset.jsonl."""
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


SHOW_ALL_MODE = False
logs = []
TOTAL_LOGS = 0
current_idx = 0


def get_logs(show_all=False):
    """Returns interaction logs. If show_all=False, only unreviewed records are returned."""
    if not os.path.exists(LOG_FILE):
        return [], 0
    reviewed_ts = get_reviewed_timestamps()
    total_count = 0
    records = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Filter out DG direct grounding logs if any are in this file
                    if data.get("mode") == "direct_grounding":
                        continue
                    total_count += 1
                    if show_all or data.get("timestamp") not in reviewed_ts:
                        records.append(data)
                except Exception:
                    pass
    return records, total_count


def resolve_image_path(record_or_path):
    """Dynamically resolves the absolute path of the target image (prefers _som.png)."""
    if isinstance(record_or_path, dict):
        img_path = record_or_path.get("annotated_image_path")
        if not img_path:
            sc_path = record_or_path.get("screenshot_path", "")
            if sc_path:
                img_path = sc_path.replace(".png", "_som.png")
    else:
        img_path = record_or_path

    if not img_path:
        return None
    if os.path.exists(img_path):
        return img_path
    
    filename = os.path.basename(img_path)
    log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
    
    for root, _, files in os.walk(log_dir):
        if filename in files:
            found = os.path.join(root, filename)
            if os.path.exists(found):
                return found
            
    parts = img_path.split("states/")
    if len(parts) > 1:
        rel_state = os.path.join("states", parts[-1])
        c1 = os.path.join(log_dir, rel_state)
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(os.path.dirname(log_dir), rel_state)
        if os.path.exists(c2):
            return c2

    return img_path if os.path.exists(img_path) else None


def calculate_statistics():
    """Calculates comprehensive error rate statistics from finetune_dataset.jsonl and interaction_log.jsonl."""
    # Keep only the latest entry per timestamp, so duplicate lines are not counted twice
    reviewed_by_ts = {}
    if os.path.exists(CLEAN_FILE):
        with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    reviewed_by_ts[data.get("timestamp") or object()] = data
    reviewed_records = list(reviewed_by_ts.values())


    # Same denominator as the review queue (DG-mode logs excluded), so percentages match the progress bar
    log_records, total_raw_logs = get_logs(show_all=True)
    # CLEAN_FILE can hold reviews from other logs in the same folder; only count the ones for this log
    log_ts = {r.get("timestamp") for r in log_records}
    reviewed_records = [r for r in reviewed_records if r.get("timestamp") in log_ts]
    total_reviewed = len(reviewed_records)

    if total_reviewed == 0:
        return f"""### 📊 Error Rate Statistics Dashboard (Dataset: `{LOG_FILE}`)
*Currently there are **{total_raw_logs}** execution logs in total. No reviewed ground-truth data available yet. Please review and save ground truth to generate metrics.*
"""

    correct_count = 0
    error_count = 0
    discarded_count = 0
    no_target_count = 0
    
    lang_stats = {}
    step_stats = {}
    task_stats = {}

    for r in reviewed_records:
        lang = r.get("language", "Unknown")
        step = r.get("step_name", "Unknown")
        
        if lang not in lang_stats:
            lang_stats[lang] = {"total": 0, "correct": 0, "error": 0}
        if step not in step_stats:
            step_stats[step] = {"total": 0, "correct": 0, "error": 0}

        if r.get("discarded", False) or r.get("ground_truth_id") == -1:
            discarded_count += 1
            continue

        ai_id = r.get("target_id", -1)
        gt_id = r.get("ground_truth_id", -1)

        if ai_id == -1 or ai_id is None:
            no_target_count += 1

        lang_stats[lang]["total"] += 1
        step_stats[step]["total"] += 1
        # Repeated rounds of the same task are not independent samples, so track outcomes per task
        task_stats.setdefault((lang, step, r.get("instruction", "")), []).append(ai_id == gt_id and ai_id != -1)

        if ai_id == gt_id and ai_id != -1:
            correct_count += 1
            lang_stats[lang]["correct"] += 1
            step_stats[step]["correct"] += 1
        else:
            error_count += 1
            lang_stats[lang]["error"] += 1
            step_stats[step]["error"] += 1

    valid_reviewed = total_reviewed - discarded_count
    acc_rate = (correct_count / valid_reviewed * 100) if valid_reviewed > 0 else 0.0
    err_rate = (error_count / valid_reviewed * 100) if valid_reviewed > 0 else 0.0
    distinct_tasks = len(task_stats)
    repeats_per_task = (valid_reviewed / distinct_tasks) if distinct_tasks else 0.0
    inconsistent_tasks = sum(1 for outcomes in task_stats.values() if len(set(outcomes)) > 1)
    # Each task counts once: mean of the per-task accuracies instead of one vote per repeated sample
    task_level_rate = (sum(sum(o) / len(o) for o in task_stats.values()) / distinct_tasks * 100) if distinct_tasks else 0.0
    run_count = len({r.get("run_id") for r in reviewed_records if r.get("run_id")})

    report_md = f"""### 📊 Experiment Metrics & Error Rate Dashboard
> **Current Log File**: `{LOG_FILE}`  
> **Clean Dataset Export Path**: `{CLEAN_FILE}`

| Statistical Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Execution Logs** | {total_raw_logs} entries | 100% |
| **Reviewed Samples** | {total_reviewed} entries | {(total_reviewed/total_raw_logs*100 if total_raw_logs else 0):.1f}% |
| **Valid Evaluation Samples** | {valid_reviewed} entries | {(valid_reviewed/total_reviewed*100 if total_reviewed else 0):.1f}% |
| **✅ AI Prediction Correct** | {correct_count} entries | **{acc_rate:.2f}%** |
| **❌ AI Prediction Error** | {error_count} entries | **{err_rate:.2f}%** |
| **⚠️ AI No Target / Fallback** | {no_target_count} entries | - |
| **🧩 Distinct Tasks (language + step + instruction)** | {distinct_tasks} tasks | {repeats_per_task:.1f} samples per task |
| **⚖️ Task-level Accuracy (each task counted once)** | - | **{task_level_rate:.2f}%** |
| **🔀 Inconsistent Tasks (outcome differs between repeats)** | {inconsistent_tasks} tasks | - |
| **🏷️ Runs (records with run_id)** | {run_count} runs | - |
| **🗑️ Discarded Samples** | {discarded_count} entries | - |

#### 🌐 Error Rate Breakdown by Language

| Language | Valid Samples | Correct | Error | **Error Rate** | **Accuracy** |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for lang, s in lang_stats.items():
        if s["total"] > 0:
            l_err = s["error"] / s["total"] * 100
            l_acc = s["correct"] / s["total"] * 100
            report_md += f"| {lang} | {s['total']} | {s['correct']} | {s['error']} | **{l_err:.2f}%** | {l_acc:.2f}% |\n"

    report_md += """
#### 📋 Error Rate Breakdown by Interaction Step

| Step Name | Valid Samples | Correct | Error | **Error Rate** |
| :--- | :--- | :--- | :--- | :--- |
"""
    for step, s in step_stats.items():
        if s["total"] > 0:
            st_err = s["error"] / s["total"] * 100
            report_md += f"| `{step}` | {s['total']} | {s['correct']} | {s['error']} | **{st_err:.2f}%** |\n"

    return report_md


def switch_dataset(selected_option, show_all_choice):
    global LOG_FILE, CLEAN_FILE, logs, TOTAL_LOGS, current_idx
    if selected_option in DATASET_OPTIONS:
        LOG_FILE = DATASET_OPTIONS[selected_option]
    else:
        LOG_FILE = selected_option
    CLEAN_FILE = os.path.join(os.path.dirname(LOG_FILE), "finetune_dataset.jsonl")
    current_idx = 0
    show_all = (show_all_choice == "Show all entries (including reviewed)")
    logs, TOTAL_LOGS = get_logs(show_all=show_all)
    slider_max = max(2, len(logs))
    
    img_path, instruction, ai_choice, default_id = load_record(0, show_all)
    stats_md = calculate_statistics()
    return img_path, instruction, ai_choice, default_id, stats_md, gr.update(maximum=slider_max, value=1, interactive=(len(logs) > 1))


def load_record(idx, show_all=False):
    global logs, TOTAL_LOGS, current_idx
    logs, TOTAL_LOGS = get_logs(show_all=show_all)
    current_idx = idx
    
    reviewed_ts = get_reviewed_timestamps()
    reviewed_count = len(reviewed_ts)
    
    if not logs or idx >= len(logs):
        if show_all:
            progress_info = f"⚠️ No entries found in `{LOG_FILE}`."
        else:
            progress_info = f"🎉 All entries in the current dataset have been reviewed! (Progress: {reviewed_count}/{TOTAL_LOGS})\n\n💡 *Tip: Switch to 'Show all entries' above to inspect or modify existing annotations.*"
        return None, progress_info, "", ""
    
    record = logs[idx]
    img_path = resolve_image_path(record)
    
    mode_tag = " [All Entries Mode]" if show_all else " [Unreviewed Only]"
    progress_info = f"### 📊 Review Progress{mode_tag}: Record {idx + 1} of {len(logs)} | Total Reviewed: {reviewed_count}/{TOTAL_LOGS}\n"
    instruction = f"{progress_info}### 📋 Task Instruction: `{record.get('instruction', '')}`"
    if not img_path:
        raw_img = record.get('annotated_image_path') or record.get('screenshot_path') or 'Unknown'
        instruction += f"\n\n⚠️ **Note**: Screenshot image file not found on disk (`{raw_img}`)."
    
    target_id = record.get('target_id', -1)
    confidence = record.get('decision', {}).get('confidence', 0.0)
    existing_gt = record.get('ground_truth_id', None)
    
    gt_info = f" | Existing GT ID: `[{existing_gt}]`" if existing_gt is not None else ""
    if target_id == -1:
        ai_choice = f"🚨 **AI could not locate target (target_id = -1)** | App Locale: {record.get('language')}{gt_info}"
        default_id = str(existing_gt) if existing_gt is not None else ""
    else:
        ai_choice = f"🤖 **AI Selected ID**: `[{target_id}]` (Confidence: {confidence:.2f}) | App Locale: {record.get('language')}{gt_info}"
        default_id = str(existing_gt) if existing_gt is not None else str(target_id)
        
    return img_path, instruction, ai_choice, default_id


def save_record_to_clean_file(record):
    """Saves a record idempotently to CLEAN_FILE (replaces the matching timestamp, otherwise appends)."""
    os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
    existing_lines = []
    replaced = False
    target_ts = record.get("timestamp")
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


def save_and_next(human_corrected_id, show_all_choice):
    global current_idx, logs
    show_all = (show_all_choice == "Show all entries (including reviewed)")
    if current_idx < len(logs):
        record = logs[current_idx]
        val = human_corrected_id.strip() if human_corrected_id else str(record.get("target_id", -1))
        try:
            record["ground_truth_id"] = int(val)
        except ValueError:
            record["ground_truth_id"] = record.get("target_id", -1)

        save_record_to_clean_file(record)
            
        if show_all:
            current_idx = min(current_idx + 1, len(logs) - 1)
        else:
            current_idx = 0
            
    img_path, instruction, ai_choice, default_id = load_record(current_idx, show_all)
    stats_md = calculate_statistics()
    slider_val = min(current_idx + 1, max(1, len(logs)))
    return img_path, instruction, ai_choice, default_id, stats_md, gr.update(maximum=max(2, len(logs)), value=slider_val, interactive=(len(logs) > 1))


def discard_and_next(show_all_choice):
    global current_idx, logs
    show_all = (show_all_choice == "Show all entries (including reviewed)")
    if current_idx < len(logs):
        record = logs[current_idx]
        record["ground_truth_id"] = -1
        record["discarded"] = True
        save_record_to_clean_file(record)
            
        if show_all:
            current_idx = min(current_idx + 1, len(logs) - 1)
        else:
            current_idx = 0
            
    img_path, instruction, ai_choice, default_id = load_record(current_idx, show_all)
    stats_md = calculate_statistics()
    slider_val = min(current_idx + 1, max(1, len(logs)))
    return img_path, instruction, ai_choice, default_id, stats_md, gr.update(maximum=max(2, len(logs)), value=slider_val, interactive=(len(logs) > 1))


def go_back(show_all_choice):
    global current_idx, logs
    show_all = (show_all_choice == "Show all entries (including reviewed)")
    if show_all:
        current_idx = max(0, current_idx - 1)
    else:
        if os.path.exists(CLEAN_FILE):
            with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if lines:
                with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
                    f.writelines(lines[:-1])
                print(f"⬅️ Reverted last decision in {CLEAN_FILE}")
        current_idx = 0
            
    img_path, instruction, ai_choice, default_id = load_record(current_idx, show_all)
    stats_md = calculate_statistics()
    slider_val = min(current_idx + 1, max(1, len(logs)))
    return img_path, instruction, ai_choice, default_id, stats_md, gr.update(maximum=max(2, len(logs)), value=slider_val, interactive=(len(logs) > 1))


def jump_to_record(idx_1based, show_all_choice):
    show_all = (show_all_choice == "Show all entries (including reviewed)")
    idx = max(0, int(idx_1based) - 1)
    img_path, instruction, ai_choice, default_id = load_record(idx, show_all)
    stats_md = calculate_statistics()
    return img_path, instruction, ai_choice, default_id, stats_md


def toggle_filter_mode(mode_choice):
    show_all = (mode_choice == "Show all entries (including reviewed)")
    img_path, instruction, ai_choice, default_id = load_record(0, show_all)
    stats_md = calculate_statistics()
    slider_max = max(2, len(logs))
    return img_path, instruction, ai_choice, default_id, stats_md, gr.update(maximum=slider_max, value=1, interactive=(len(logs) > 1))


initial_label = list(DATASET_OPTIONS.keys())[0]
for k, v in DATASET_OPTIONS.items():
    if v == LOG_FILE:
        initial_label = k
        break

# Web UI Layout
with gr.Blocks(title="VGT Data Flywheel - Review & Error Rate Console") as demo:
    gr.Markdown("## 🕵️‍♂️ VGT Data Flywheel - Human Review & Error Rate Console (SoM Box Review)")
    
    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            choices=list(DATASET_OPTIONS.keys()),
            value=initial_label,
            label="📁 Select Model / Experiment Group (Dataset Selection)",
            interactive=True,
            scale=3
        )
        filter_mode_radio = gr.Radio(
            choices=["Show unreviewed only", "Show all entries (including reviewed)"],
            value="Show all entries (including reviewed)",
            label="👁️ Display Mode (Filter)",
            interactive=True,
            scale=2
        )
    
    with gr.Tab("🔍 Interaction Review & Ground Truth Annotation"):
        with gr.Row():
            init_count = len(get_logs(show_all=True)[0])
            index_slider = gr.Slider(
                minimum=1,
                maximum=max(2, init_count),
                value=1,
                step=1,
                label="🔢 Jump to Record (Index Navigator)",
                interactive=(init_count > 1)
            )
        with gr.Row():
            image_display = gr.Image(type="filepath", interactive=False)
            with gr.Column():
                instruction_text = gr.Markdown()
                ai_choice_text = gr.Markdown()
                human_input = gr.Textbox(label="Please enter the correct Element ID (press Enter/Save if AI prediction is correct)")
                with gr.Row():
                    submit_btn = gr.Button("💾 Save Ground Truth & Next", variant="primary")
                    discard_btn = gr.Button("🗑️ Discard Record", variant="stop")
                    prev_btn = gr.Button("⬅️ Previous Record / Back", variant="secondary")

    with gr.Tab("📊 Real-time Error Rate & Metrics Dashboard"):
        stats_display = gr.Markdown(value=calculate_statistics())
        refresh_stats_btn = gr.Button("🔄 Recalculate & Refresh Statistics", variant="secondary")

    # Event handlers
    dataset_dropdown.change(
        fn=switch_dataset,
        inputs=[dataset_dropdown, filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )
    
    filter_mode_radio.change(
        fn=toggle_filter_mode,
        inputs=[filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )

    index_slider.change(
        fn=jump_to_record,
        inputs=[index_slider, filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display]
    )
    
    submit_btn.click(
        fn=save_and_next,
        inputs=[human_input, filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )
    
    discard_btn.click(
        fn=discard_and_next,
        inputs=[filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )
    
    prev_btn.click(
        fn=go_back,
        inputs=[filter_mode_radio],
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )
    
    refresh_stats_btn.click(
        fn=calculate_statistics,
        outputs=[stats_display]
    )

    demo.load(
        fn=lambda: load_record(0, show_all=True) + (calculate_statistics(), gr.update(maximum=max(2, len(get_logs(show_all=True)[0])), value=1, interactive=(len(get_logs(show_all=True)[0]) > 1))),
        outputs=[image_display, instruction_text, ai_choice_text, human_input, stats_display, index_slider]
    )


if __name__ == "__main__":
    from pathlib import Path
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    home_dir = str(Path.home())
    allowed = [home_dir, "/Users/taylorzhuang", "/Users/taylorzhuang/social-app", "/Users/taylorzhuang/Masterarbeit"]
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=False, allowed_paths=allowed)
    except OSError:
        demo.launch(server_name="127.0.0.1", share=False, allowed_paths=allowed)