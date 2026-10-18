# 📱 Auto-Screen: Universal Visual GUI Automation & Data Flywheel Suite

**Auto-Screen** is a universal framework for mobile (iOS) visual GUI automated testing, multilingual interface regression comparison, and continuous data flywheel fine-tuning.

The system supports two visual agent inference modes:
1. **Set-of-Mark (SoM)**: Visual bounding box annotations and numbered marks.
2. **Direct Grounding (DG)**: End-to-end continuous coordinate prediction.

Auto-Screen seamlessly evaluates locally fine-tuned models (e.g. Qwen3-VL) as well as cloud vision models (Claude 3.5 Sonnet, GPT-4o).

---

## 🏗️ System Architecture

```
Auto-Screen/
├── auto_screen.py         # 🚀 Unified top-level CLI entrypoint
├── generic_engine.py      # ⚙️ Declarative test execution engine (Appium XCUITest)
├── review_tool.py         # 🔍 Set-of-Mark (SoM) manual review & evaluation console (Gradio)
├── review_tool_DG.py      # 🎯 Direct Grounding (DG) review & coordinate evaluation console
├── compare_images.py      # 📊 Multi-algorithm visual regression suite (Pixel / OpenCV / VLM)
│
├── configs/               # 📁 Declarative application configurations
│   ├── template_app.json  # 💡 Template for integrating new applications
│   ├── bluesky.json       # Bluesky test flows & environment configuration
│   └── katapp.json        # Katapp test flows & environment configuration
│
├── common_agent/          # 🧠 Unified Agent backend engine & CV foundation
│   ├── agent_facade.py    # Multi-provider agent facade (Claude, OpenAI, Qwen)
│   ├── base_cv.py         # Core computer vision & heuristic logic
│   └── model_client.py    # Unified API model client wrapper
│
├── Qwen_trained/          # 🤖 Qwen3-VL locally fine-tuned agent
├── Qwen_untrain/          # 🤖 Qwen3-VL zero-shot baseline agent
├── claude/                # 🤖 Claude 3.5 Sonnet cloud vision baseline
├── openAI_6.0_Luna/       # 🤖 GPT-4o-mini baseline
├── openAI_6.0_Sol/        # 🤖 GPT-4o baseline
│
├── tools/                 # 🛠️ Utilities
│   ├── prepare_dataset.py # Export reviewed data for fine-tuning
│   └── swift_ocr.swift    # Native Apple Vision high-precision OCR extractor
└── run_pipeline_local.sh  # 🔄 Local one-click visual regression pipeline
```

---

## 🚀 Quick Start

### 1. Overview and Available Configurations
```bash
python3 auto_screen.py list
```

### 2. Prerequisites
Before executing tests, ensure the background services are active:

1. **Launch Appium Server**:
   ```bash
   appium
   ```
   *Listens on `http://localhost:4723` by default.*

2. **Boot iOS Simulator** (Appium will attach to the booted simulator):
   ```bash
   # List available simulators
   xcrun simctl list devices | grep Booted

   # Or boot a simulator (e.g. iPhone 15 Pro):
   xcrun simctl boot "iPhone 15 Pro"
   ```

### 3. Run Automated Tests
```bash
# Run Bluesky test with fine-tuned Qwen using Direct Grounding mode
python3 auto_screen.py test --app bluesky --model Qwen_trained --mode dg

# Run Katapp test with Claude
python3 auto_screen.py test --app katapp --model claude

# Run with custom configuration
python3 auto_screen.py test --config configs/my_app.json --loops 1
```

### 4. Interactive Review & Annotation Console
```bash
# Launch Direct Grounding review console
python3 auto_screen.py review --mode dg --port 7860

# Launch Set-of-Mark review console
python3 auto_screen.py review --mode som --port 7861
```

### 5. Visual Regression Image Comparison
```bash
python3 auto_screen.py compare --baseline ./baseline_dir --actual ./actual_dir --output ./diff_reports
```

### 6. Export Fine-Tuning Dataset (Data Flywheel)
```bash
python3 auto_screen.py dataset --input Qwen_trained/diff_results/finetune_dataset_DG.jsonl --output diff_results/qwen_vl_dataset.json
```

---

## 📲 Integrating Any New App (3 Steps)

### Step 1: Build Simulator App (.app)
Compile an iOS Simulator build in Xcode or Flutter / React Native:
```bash
# Flutter
flutter build ios --simulator
```
The output is typically located at `build/ios/iphonesimulator/Runner.app`.

### Step 2: Create Declarative Configuration
Copy `configs/template_app.json` to `configs/your_app.json`:
```json
{
  "app_config": {
    "platform_name": "iOS",
    "platform_version": "17.5",
    "device_name": "iPhone 15 Pro",
    "app_path": "/path/to/YourApp.app",
    "bundle_id": "com.company.yourapp",
    "appium_url": "http://localhost:4723"
  },
  "initial_setup_steps": [
    {
      "step_name": "click_login",
      "instruction": "Click the button labeled 'Sign In'",
      "action": "click"
    }
  ],
  "test_flows": [
    {
      "folder": "baseline_EN",
      "language_name": "English",
      "steps": [
        {
          "step_name": "go_to_profile",
          "instruction": "Click the user avatar in the bottom navigation bar",
          "action": "click"
        }
      ]
    }
  ]
}
```

### Step 3: Run and Review
```bash
python3 auto_screen.py test --app your_app --model Qwen_trained --mode dg
python3 auto_screen.py review --mode dg
```
Review model predictions in the Web UI, save human calibrations, and feed corrections directly back into the fine-tuning dataset for continuous improvement!