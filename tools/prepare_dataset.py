import os
import json
from pathlib import Path

# Path configuration for Data Flywheel
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "diff_results" / "finetune_dataset.jsonl"
OUTPUT_FILE = BASE_DIR / "diff_results" / "qwen_vl_dataset.json"

# Fallback to current directory if not found in diff_results
if not INPUT_FILE.exists() and Path("finetune_dataset.jsonl").exists():
    INPUT_FILE = Path("finetune_dataset.jsonl")

def convert_to_qwen_dataset(input_path, output_path):
    """
    Converts manually reviewed Ground Truth records (finetune_dataset.jsonl)
    into LLaMA-Factory / Qwen-VL ShareGPT format for SFT fine-tuning.
    """
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_path}' not found.")
        return

    llama_factory_data = []
    skipped_discarded = 0
    skipped_missing_img = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            
            # 1. Skip discarded or invalid entries (ground_truth_id = -1)
            ground_truth = record.get("ground_truth_id", -1)
            is_discarded = record.get("discarded", False)
            if is_discarded or ground_truth == -1 or ground_truth is None:
                skipped_discarded += 1
                continue
            
            # 2. Extract image path and instruction
            instruction = record.get("instruction", "")
            img_path = record.get("annotated_image_path", "")
            
            # Check if image file exists on disk
            if not os.path.exists(img_path):
                skipped_missing_img += 1
                continue
            
            # 3. Format for LLaMA-Factory / Qwen-VL ShareGPT format
            formatted_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>This is a UI testing task. Locate the UI element that matches this instruction: '{instruction}'. Output the target ID in JSON format: {{\"target_id\": X}}."
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"target_id": int(ground_truth)})
                    }
                ],
                "images": [
                    img_path
                ]
            }
            llama_factory_data.append(formatted_record)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(llama_factory_data, out_f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("🚀 Qwen-VL fine-tuning dataset successfully generated!")
    print(f"📂 Input file: {input_path}")
    print(f"📂 Output file: {output_path}")
    print(f"✅ Valid SFT records: {len(llama_factory_data)}")
    print(f"🗑️ Skipped discarded/invalid records: {skipped_discarded}")
    if skipped_missing_img > 0:
        print(f"⚠️ Skipped missing image files: {skipped_missing_img}")
    print("=" * 60)

if __name__ == "__main__":
    convert_to_qwen_dataset(INPUT_FILE, OUTPUT_FILE)