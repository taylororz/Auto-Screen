# Auto-Screen

Auto-Screen is a visual regression testing framework for multilingual iOS applications, developed as the research artifact of a Master's thesis (M.Sc. Wirtschaftsinformatik, Reutlingen University, 2026).

Auto-Screen drives iOS apps in the Simulator via Appium (XCUITest) across configured languages. A vision-language model (VLM) localizes UI elements directly from screenshots without relying on the accessibility tree or element IDs, allowing a single test flow definition to run across multiple languages. Captured screenshots are compared using OpenCV/NumPy image processing, and visual differences are summarized in an interactive HTML report.

Two comparison modes are supported:
- **Language comparison**: compares screenshots of a reference language against other languages from the same test run.
- **Version comparison**: compares screenshots of the same language folder across two app builds (used for defect detection and CI).

Evaluated repository states are tagged as `eval-final` (`b119068`, 2026-10-20) and `eval-final-2` (`2d2693e`, 2026-10-28).

---

## Perception Modes

| Mode | Script | Model Input | Model Output |
|---|---|---|---|
| Set-of-Mark (SoM), `--mode som` | `visual_agent.py` | Screenshot annotated with numbered red bounding boxes extracted via OpenCV (Canny edges, adaptive threshold, contour filtering/merging) | `{"target_id": <id>}` |
| Direct Grounding (DG), `--mode dg` | `visual_agent_DG.py` | Clean unmarked screenshot | Target coordinates (`[0, 1000]` normalized grid for Qwen models; pixel coordinates for cloud models) |

Both agent modules expose the same interface to `generic_engine.py`.

## Models

| Directory | Model | Role |
|---|---|---|
| `Qwen_trained/` | Qwen3-VL-4B-Instruct (LoRA fine-tuned, weights in `Qwen_trained/Qwen3-VL-4B-V2/`) | Default local model used by the pipeline (MPS/CUDA/CPU, `bfloat16` on GPU) |
| `Qwen_untrain/` | Qwen3-VL-4B-Instruct base model (`BASE_MODEL_NAME`) | Untrained local baseline |
| `claude/` | Claude Sonnet 5.5 (`CLAUDE_AGENT_MODEL`, default `claude-sonnet-5-5`) | Cloud baseline |
| `openAI_6.0_Luna/` | GPT-6 Luna (`OPENAI_AGENT_MODEL`, default `gpt-6-luna`) | Cloud baseline |
| `openAI_6.0_Sol/` | GPT-6 Sol (`OPENAI_AGENT_MODEL`, default `gpt-6-sol`) | Cloud baseline |

---

## Repository Layout

```text
Auto-Screen/
├── auto_screen.py              # Unified CLI (list, test, review, compare, dataset)
├── generic_engine.py           # Config-driven Appium/XCUITest test execution engine
├── compare_images.py           # Pixel & grayscale image comparison and HTML/JSON reporting
├── review_tool.py              # Gradio review tool for SoM interactions
├── review_tool_DG.py           # Gradio review tool and SFT exporter for DG interactions
├── assert_pipeline_result.py   # CI quality gate assertion script
├── run_pipeline_local.sh       # End-to-end local/CI execution script
├── configs/                    # App configurations (katapp.json, bluesky.json, simplenote.json, template_app.json)
├── common_agent/               # Shared CV detection, VLM clients, agent facade, and local diff triage
├── Qwen_trained/               # Fine-tuned local Qwen3-VL agent (SoM & DG)
├── Qwen_untrain/               # Base local Qwen3-VL agent (SoM & DG)
├── claude/                     # Claude visual agent (SoM & DG)
├── openAI_6.0_Luna/            # OpenAI GPT-6 Luna visual agent (SoM & DG)
├── openAI_6.0_Sol/             # OpenAI GPT-6 Sol visual agent (SoM & DG)
├── tools/                      # Dataset, baseline, calibration, and matrix utilities
│   ├── prepare_dataset.py
│   ├── convert_ios1k_dataset.py
│   ├── swift_ocr.swift
│   ├── ios1k_cluster_bootstrap.py
│   ├── approve_baseline.py
│   ├── calibrate_thresholds.py
│   └── run_matrix.py
├── ios1k_som_eval.py           # FF2: SoM benchmark on the cleaned iOS-1K evaluation split
├── results/                    # Benchmark outputs (base.jsonl, lora.jsonl, summary.json, e16/)
├── ff3/                        # FF3: Seeded-defect evaluation suite and analysis scripts
│   ├── ff3_eval.py
│   ├── ff3_shift_analysis.py
│   ├── bbox_picker.py
│   ├── switch_app.sh
│   ├── seeded_defects.json
│   ├── FF3_Protokoll.md
│   ├── FF3_steps.md
│   └── ci_check/
├── eval_ext/                   # Supplemental evaluation scripts and results (E1–E16, PROTOKOLL.md)
└── .github/workflows/          # GitHub Actions workflows (visual_pipeline.yml, approve_baseline.yml)
```

---

## Requirements

- **macOS** with Xcode and iOS Simulator (evaluated with Xcode 27.0, iPhone 18 Pro, iOS 27.0 on Apple M5 Pro, 24 GB RAM).
- **Python 3.10+** (evaluated with 3.10.11).
- **Appium 3** with XCUITest driver (evaluated with Appium 3.5.0):
  ```bash
  npm install -g appium
  appium driver install xcuitest
  ```
- **Python dependencies**:
  ```bash
  pip install -r requirements.txt
  pip install gradio  # required for review_tool.py and review_tool_DG.py
  ```
- **Cloud API keys** (when using cloud baselines): `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

---

## Usage

### 1. Start Appium and iOS Simulator

```bash
appium
xcrun simctl boot "iPhone 18 Pro"
```

### 2. Run UI Tests

```bash
python3 auto_screen.py list
python3 auto_screen.py test --app katapp --model Qwen_trained --mode dg
python3 auto_screen.py test --app bluesky --model claude --mode som --loops 2
python3 auto_screen.py test --app simplenote --model Qwen_trained --mode dg --loops 5
```

- `--loops / -l`: number of execution rounds.
- `--no-loop-reset`: skip app reset between loops.
- Image resolution controls:
  - Cloud models (`common_agent/model_client.py`): `MAX_IMAGE_EDGE` sets the maximum longest edge (default `1600`; `0` keeps original resolution).
  - Local Qwen model (`Qwen_trained/`): `LOCAL_MAX_IMAGE_EDGE` (default `1600` for SoM, `1280` for DG; `0` keeps original resolution) and `LOCAL_DG_MAX_PIXELS` (default `1024 * 28 * 28`).
- Interaction logs are written to `<model>/diff_results/interaction_log.jsonl` (SoM) or `interaction_log_DG.jsonl` (DG).

### 3. Review Interactions

```bash
python3 auto_screen.py review --mode som --port 7861
python3 auto_screen.py review --mode dg  --port 7860
```

Reviewed records are saved to `finetune_dataset.jsonl` (SoM) and `finetune_dataset_DG.jsonl` (DG, 40 px hit tolerance).

### 4. Compare Screenshots

```bash
# Language comparison (reference language vs. other languages in the same run)
python3 compare_images.py --batch --app katapp --model Qwen_trained

# Folder pair comparison
python3 compare_images.py --batch --app katapp --model Qwen_trained \
    --baseline-folder <folder_A> --actual-folders <folder_B>

# Version comparison (each language folder vs. matching folder of a previous build)
python3 compare_images.py --batch --app katapp --model Qwen_trained \
    --version-baseline-root <previous_screenshots_dir> \
    --output-root diff_results/version_comparison \
    --baseline-label <commit_A> --actual-label <commit_B>
```

Additional flags: `--pages`, `--flow`, `--only-diffs`, `--threshold` (default `38`), `--min-box-size` (default `14`), `--mask-top` (default `140`), `--mask-bottom` (default `70`), `--ai` (OpenAI triage), `--ai-local` (local Qwen3-VL triage).

Outputs are saved to `summary_report.json` and `summary_report.html`.

### 5. Export Training Data

```bash
python3 tools/prepare_dataset.py
```

Converts `diff_results/finetune_dataset.jsonl` into LLaMA-Factory format (`diff_results/qwen_vl_dataset.json`).

---

## Comparison Methods and Default Thresholds

Both comparison methods mask the top 140 px (status bar) and bottom 70 px (home indicator) and mark pixels whose difference exceeds $\tau = 38$:
- **Pixel comparison**: maximum absolute difference across RGB channels.
- **Grayscale comparison**: absolute luminance difference after grayscale conversion.

Detected pixels undergo morphological closing ($21 \times 7$), dilation ($5 \times 5$), nearby-box merging (18 px horizontal, 10 px vertical), and noise filtering (minimum $14 \times 14$ px, area $\ge 150\text{ px}^2$, $\ge 25$ changed pixels, $\ge 4\%$ density).

| Parameter | Default | Component |
|---|---|---|
| Confidence threshold | `0.35` (`min_confidence`) | Step check 1 |
| Page change threshold | Gray diff $> 20$ on $> 0.5\%$ of pixels | Step check 2 |
| Pixel/grayscale threshold $\tau$ | `38` | `compare_images.py` |
| Status bar / home indicator masks | `140 px` top, `70 px` bottom | `compare_images.py` |
| DG target tolerance | `40 px` | `review_tool_DG.py` / evaluation |

---

## CI/CD Pipeline

1. **Application build workflow** (`build_and_trigger_test.yml` in the app repository): builds the iOS Simulator bundle (`Runner.app`), uploads it as an artifact, and triggers `repository_dispatch` (`katapp_build_completed`).
2. **Visual regression workflow** ([`.github/workflows/visual_pipeline.yml`](file:///Users/taylorzhuang/Auto-Screen/.github/workflows/visual_pipeline.yml)): runs on a self-hosted macOS runner, downloads the app build and previous baseline screenshots, executes [`run_pipeline_local.sh`](file:///Users/taylorzhuang/Auto-Screen/run_pipeline_local.sh) (DG test run, language comparison, version comparison, and [`assert_pipeline_result.py`](file:///Users/taylorzhuang/Auto-Screen/assert_pipeline_result.py)), and uploads the visual regression report and updated baseline screenshots.
3. **Baseline approval workflow** ([`.github/workflows/approve_baseline.yml`](file:///Users/taylorzhuang/Auto-Screen/.github/workflows/approve_baseline.yml)): pins a verified CI run artifact as `approved-baseline-<app>` when `BASELINE_MODE=approved`.

---

## Adding an App

Copy [`configs/template_app.json`](file:///Users/taylorzhuang/Auto-Screen/configs/template_app.json) to `configs/<app>.json` and configure `app_config`, `initial_setup_steps`, and per-language `test_flows`.

Supported step properties:
- `action`: `click` (default), `type_text`, or scroll/swipe actions
- `page_name`: report page identifier (defaults to `step_name`)
- `min_confidence`: step confidence threshold (default `0.35`)
- `optional`: skip step if target is not found
- `skip_layout_check`: bypass post-step layout change verification
- `wait_before`, `wait_after`: delay in seconds (defaults `0` and `1.5`)
- `clear`, `hide_keyboard`, `is_password`: text input options (`text` supports `$ENV_VAR` / `[ENV_VAR]`)
- `skip_screenshot`, `skip_log`: omit baseline screenshot or interaction log entry

---

## Optional Extensions and Supplemental Evaluations (`eval_ext/`)

Post-evaluation extensions (commit `2f50706`, disabled by default) and supplemental evaluation scripts E1–E16 ([`eval_ext/PROTOKOLL.md`](file:///Users/taylorzhuang/Auto-Screen/eval_ext/PROTOKOLL.md)):

- **Threshold sensitivity (`E1`)**: [`tools/calibrate_thresholds.py`](file:///Users/taylorzhuang/Auto-Screen/tools/calibrate_thresholds.py) evaluates $\tau$ and minimum box size grids on FF3 runs.
- **Local VLM diff triage (`E2`)**: `--ai-local` (`LOCAL_TRIAGE=1`) rates grayscale diff crops locally via [`common_agent/diff_triage.py`](file:///Users/taylorzhuang/Auto-Screen/common_agent/diff_triage.py) (`eval_ext/e2_triage.py`).
- **Sub-image refinement & tolerance (`E3`, `E3b`, `E3c`)**: `DG_SUBIMAGE_REFINE=1` runs a second-pass half-screen crop prediction in [`Qwen_trained/visual_agent_DG.py`](file:///Users/taylorzhuang/Auto-Screen/Qwen_trained/visual_agent_DG.py) (`eval_ext/e3_dg_subimage.py`, `eval_ext/e3_boxcheck.py`, `eval_ext/e3b_dg_flows.py`, `eval_ext/e3c_toleranz.py`).
- **Approved baseline management (`E4`)**: [`tools/approve_baseline.py`](file:///Users/taylorzhuang/Auto-Screen/tools/approve_baseline.py) and `BASELINE_MODE=approved` pin a verified screenshot set (`eval_ext/e4_baseline_replay.py`).
- **Screenshot stabilization & ignore regions**: top-level config keys `stability` (`hide_keyboard`, `wait_for_idle`) and `compare` (`mask_top`, `mask_bottom`, `ignore_regions`).
- **Device/appearance/locale matrix**: [`tools/run_matrix.py`](file:///Users/taylorzhuang/Auto-Screen/tools/run_matrix.py) runs combinations of simulator devices, light/dark appearance, Dynamic Type sizes, and `launch_language` / `launch_locale`.
- **Follow-up analyses (`E5`–`E16`)**:
  - `E5`–`E5i`: SoM replay reproducibility (`e5d_som_replay.py`), paired base-vs-LoRA comparison (`e5e_compare.py`), DG resolution verification (`e5g_dg_imagesize.py`), box ID permutation tests (`e5h_renumber.py`), and cloud SoM checks (`e5i_cloud_som.py`).
  - `E6`: Native resolution vs. downscaled inference for local SoM and DG (`e6_dg_fullsize.py`).
  - `E7`–`E13`: iOS-1K box provenance (`e7_boxherkunft.py`), instructions without positional hints (`e8_lagehinweise.py`), base model DG on iOS-1K (`e9_dg_base_ios1k.py`), cross-language ranking (`e10_sprachvergleich.py`), blind second-rater review (`e11_zweitbewertung.py`), target size distribution (`e12_zielgroesse.py`), and icon miss analysis (`e13_symbole.py`).
  - `E14`–`E16`: Semi-automatic cloud step verification (`e14_cloud_schritte.py`), third-app generalization on Simplenote (`e15_simplenote_review.py`), and full matrix replication (`e16_laeufe.sh`, `e16_schritte.py`).
