# Auto-Screen

Auto-Screen is a prototype for visual regression testing of multilingual iOS apps. It was built as the artifact of a master's thesis (M.Sc. Wirtschaftsinformatik, Hochschule Reutlingen, 2026).

Auto-Screen drives an app in the iOS Simulator through Appium (XCUITest) for every configured language. A vision-language model (VLM) chooses the next UI element from the screenshot alone, without the accessibility tree or element IDs, so the same test flow works across languages. Each reached view is saved as a screenshot. Screenshots are then compared with classical image processing (OpenCV/NumPy), and an HTML report shows the differences. Whether a difference is a defect is decided by a person.

Two comparisons are supported:

- **Language comparison**: the views of the reference language against the same views in the other languages of one run.
- **Version comparison**: the same language folder of two builds. This is the comparison intended for defect detection and the one used by the CI pipeline.

The evaluated state is tagged `eval-final` (commit `b119068`, 20.10.2026).

---

## Perception modes

| Mode | Script | What the model receives | What the model returns |
|---|---|---|---|
| Set-of-Mark (SoM), `--mode som` | `visual_agent.py` | Screenshot with numbered red boxes. Boxes come from classical image processing (Canny edges, adaptive threshold, merging and filtering), not from a learned detector | `{"target_id": <id>}` |
| Direct Grounding (DG), `--mode dg` | `visual_agent_DG.py` | Unmarked screenshot | A point. Qwen models answer in the normalized grid `[0, 1000]`, the cloud models in pixels |

Both modules offer the same interface, so the engine is identical for both modes.

## Models

| Folder | Model | Role |
|---|---|---|
| `Qwen_trained/` | Qwen3-VL-4B-Instruct with LoRA, weights in `Qwen_trained/Qwen3-VL-4B-V2/` (not in Git) | Default model, used by the pipeline. Runs locally (MPS, CUDA or CPU, bfloat16 on GPU) |
| `Qwen_untrain/` | Qwen3-VL-4B-Instruct without fine-tuning (`BASE_MODEL_NAME`) | Baseline for the evaluation |
| `claude/` | Claude Sonnet 5.5 (`claude-sonnet-5-5`, override with `CLAUDE_AGENT_MODEL`) | Cloud comparison, evaluation only |
| `openAI_6.0_Luna/` | GPT-6 Luna (`gpt-6-luna`, override with `OPENAI_AGENT_MODEL`) | Cloud comparison, evaluation only |
| `openAI_6.0_Sol/` | GPT-6 Sol (`gpt-6-sol`, override with `OPENAI_AGENT_MODEL`) | Cloud comparison, evaluation only |

The cloud models are not intended for regular operation because screenshots must not leave the company (data protection requirement).

---

## Repository layout

```
Auto-Screen/
├── auto_screen.py              # CLI: list, test, review, compare, dataset
├── generic_engine.py           # Config-driven test engine (Appium/XCUITest), step checks, screenshots, logs
├── compare_images.py           # Image comparison (pixel and grayscale), HTML/JSON report
├── review_tool.py              # Review tool for SoM steps (Gradio)
├── review_tool_DG.py           # Review tool for DG steps (Gradio), SFT export
├── assert_pipeline_result.py   # Quality gate of the pipeline
├── run_pipeline_local.sh       # Runs test, language comparison, version comparison and gate
├── configs/                    # App configurations (katapp.json, bluesky.json, template_app.json)
├── common_agent/               # Shared agent code: facade, CV functions, API clients
├── Qwen_trained/ Qwen_untrain/ claude/ openAI_6.0_Luna/ openAI_6.0_Sol/   # One folder per model
├── tools/                      # Dataset tools (prepare_dataset.py, convert_ios1k_dataset.py, swift_ocr.swift, ios1k_cluster_bootstrap.py)
├── ios1k_som_eval.py           # FF2: SoM evaluation on the cleaned iOS-1K evaluation split
├── results/                    # Results of the iOS-1K evaluation (base.jsonl, lora.jsonl, summary.json)
├── ff3/                        # FF3: seeded-defect experiment (ff3_eval.py, seeded_defects.json, switch_app.sh, ci_check/)
└── .github/workflows/visual_pipeline.yml   # CI workflow (GitHub Actions)
```

Each model folder contains `screenshots/<language folder>/<step_name>.png` (the screenshots of a run) and `diff_results/` (logs and reviewed data). Screenshots, logs, datasets and weights are excluded from Git (`.gitignore`).

---

## Requirements

- macOS with Xcode and the iOS Simulator. Evaluated with Xcode 27.0, iPhone 18 Pro, iOS 27.0 on an Apple M5 Pro with 24 GB RAM.
- Python 3.10 (evaluated with 3.10.11).
- Appium 3 with the XCUITest driver (evaluated with Appium 3.5.0):
  ```bash
  npm install -g appium
  appium driver install xcuitest
  ```
- Python packages:
  ```bash
  pip install -r requirements.txt
  pip install gradio            # only for the review tools, not in requirements.txt
  ```
- For the cloud models: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as environment variables. The code reads them from the environment and does not load `.env` by itself.

---

## Usage

### 1. Start Appium and a simulator

```bash
appium                                   # listens on http://localhost:4723
xcrun simctl boot "iPhone 18 Pro"
```

The device and iOS version must match `app_config` in the app configuration.

### 2. Run a test

```bash
python3 auto_screen.py list                                              # configs and model folders
python3 auto_screen.py test --app katapp --model Qwen_trained --mode dg  # defaults: Qwen_trained, dg, 1 loop
python3 auto_screen.py test --app bluesky --model claude --mode som --loops 2
python3 auto_screen.py test --config configs/my_app.json
```

Options: `--loops/-l` (number of rounds), `--no-loop-reset` (no app reset between rounds).

For the cloud models, `MAX_IMAGE_EDGE` sets the longest edge of the image sent to the model (default `1600`, `0` sends the original resolution). In DG mode, use the cloud models with `MAX_IMAGE_EDGE=0` (see Known limitations).

Every step is logged to `<model>/diff_results/interaction_log.jsonl` (SoM) or `interaction_log_DG.jsonl` (DG), with instruction, screenshot, model answer, tap point, run ID, round and step index.

### 3. Review the steps

```bash
python3 auto_screen.py review --mode som --port 7861
python3 auto_screen.py review --mode dg  --port 7860
```

The SoM tool stores decisions in `finetune_dataset.jsonl`, the DG tool in `finetune_dataset_DG.jsonl` (both in the model's `diff_results/`). In the DG tool, a step counts as correct if the predicted point is at most 40 px from the target point. The DG tool can export reviewed steps as an SFT dataset.

### 4. Compare screenshots

Call `compare_images.py` directly:

```bash
# Language comparison: reference language against all other language folders of one run
python3 compare_images.py --batch --app katapp --model Qwen_trained

# Any folders, e.g. the same language folder of two builds
python3 compare_images.py --batch --app katapp --model Qwen_trained \
    --baseline-folder <folder A> --actual-folders <folder B>

# Version comparison: every language folder against the same folder of a previous run
python3 compare_images.py --batch --app katapp --model Qwen_trained \
    --version-baseline-root <screenshots of the previous build> \
    --output-root diff_results/version_comparison \
    --baseline-label <commit A> --actual-label <commit B>
```

Further options: `--pages`, `--flow`, `--only-diffs`, `--threshold` (default 38), `--min-box-size` (default 14).

The results are written to `summary_report.json` and `summary_report.html`. The report counts difference boxes, not defects. The legend entry "Exact pixel diff boxes" means the comparison by the largest channel difference at threshold 38, not exact pixel equality.

`--ai` adds an assessment of the differences by an OpenAI model (`OPENAI_VISION_MODEL`, default `gpt-4.1-mini`). It is disabled in regular operation because of the data protection requirement and was not evaluated.

### 5. Export training data

```bash
python3 tools/prepare_dataset.py
```

The script converts `diff_results/finetune_dataset.jsonl` in the repository root into `diff_results/qwen_vl_dataset.json` (LLaMA-Factory format). It uses these fixed paths and ignores `--input` and `--output`.

The fine-tuned model was trained with LoRA in LLaMA-Factory on 9,665 SoM examples (`qwen_vl_dataset_combined.json`: iOS-1K images and reviewed KatApp runs, Bluesky not included). The training configuration is documented in the thesis.

---

## Comparison method and thresholds

Both comparison methods mask the top 140 px (status bar) and the bottom 70 px (home indicator) and mark pixels whose difference exceeds τ = 38.

- **Pixel comparison**: largest difference of the three color channels.
- **Grayscale comparison**: difference of the brightness after conversion to grayscale.

Afterwards, both methods use the same steps: morphological closing (21 × 7 kernel), dilation (5 × 5), merging of nearby boxes (18 px horizontal, 10 px vertical), and a minimum box size (14 × 14 px, 150 px², at least 25 changed pixels, density at least 4 %).

| Value | Default | Used in |
|---|---|---|
| Confidence threshold | 0.35 (per step: `min_confidence`) | Step check 1 |
| Page change | gray value > 20 on > 0.5 % of the pixels | Step check 2 |
| Difference threshold τ | 38 | Image comparison |
| Masked edges | 140 px top, 70 px bottom | Image comparison |
| Target point distance | 40 px | DG review tool |

The values are code defaults and were not calibrated on a dataset. τ = 38 was set on a single image pair (KatApp home page, German vs. English).

---

## CI/CD pipeline

The pipeline consists of two GitHub Actions workflows.

1. **KatApp repository** (`build_and_trigger_test.yml`): on every push to the working branch, a GitHub-hosted macOS runner builds the app for the simulator (Flutter, no signing), uploads `Runner.app` as a ZIP artifact and sends a `repository_dispatch` event (`katapp_build_completed`) with commit, branch and run number to this repository.
2. **This repository** (`.github/workflows/visual_pipeline.yml`, self-hosted macOS runner): triggered by that event, by a push to `main` or `develop`, or manually. It
   - downloads the app artifact (fallback: last successful build, then a local build) and records commit, branch, Xcode version, device and iOS version in `app_build.json`,
   - installs the dependencies and, if missing, Appium and the XCUITest driver,
   - links the local fine-tuned weights and stops if they are missing (no other model is loaded),
   - downloads the most recent screenshots of the same branch from a different build as baseline,
   - runs `run_pipeline_local.sh`: DG test with one round, language comparison, version comparison (only if the test environment of the baseline matches) and `assert_pipeline_result.py`,
   - uploads `visual-regression-report` (always) and `screenshots-<commit>` (only on success, used as baseline by later runs).

The run fails if at least one expected view was not reached (`total_missing > 0`). Differences found by the comparisons do not block the run. They are only reported. The baseline moves forward with every run, so each build is compared with the previous build, not with an approved release.

Required secret: `AUTO_SCREEN_TRIGGER_PAT` (access to the artifacts of the KatApp repository).

The local script takes these environment variables: `APP_NAME` (default `katapp`), `MODEL_NAME` (default `Qwen_trained`), `AGENT_MODE` (default `dg`), `LOCAL_MODEL_PATH`, `PREVIOUS_SCREENSHOTS`, `PREVIOUS_APP_ID`, `CURRENT_APP_ID`.

```bash
./run_pipeline_local.sh
```

---

## Adding an app

1. Build the app for the simulator, e.g. `flutter build ios --simulator` (output: `build/ios/iphonesimulator/Runner.app`).
2. Copy `configs/template_app.json` to `configs/<app>.json` and adjust it:

```json
{
  "app_config": {
    "platform_name": "iOS",
    "platform_version": "27.0",
    "device_name": "iPhone 18 Pro",
    "app_path": "/path/to/YourApp.app",
    "bundle_id": "com.example.yourapp",
    "appium_url": "http://localhost:4723"
  },
  "initial_setup_steps": [
    { "step_name": "login_field", "instruction": "Enter the token in the text field", "action": "type_text", "text": "$MY_APP_TOKEN" }
  ],
  "test_flows": [
    {
      "folder": "baseline_EN_US",
      "language_name": "English",
      "steps": [
        { "step_name": "settings_tab", "instruction": "Click the settings tab in the bottom navigation bar", "action": "click" }
      ]
    }
  ]
}
```

Each language gets its own entry in `test_flows`. Its `folder` is the screenshot folder. Instructions are written in English for all languages. Where needed, they quote the visible label in the target language (e.g. `'事件'`).

If `text` names an environment variable (`$NAME` or `[NAME]`), the engine types the value of that variable. Use this for tokens and passwords instead of writing them into the config.

Optional step fields:

| Field | Effect |
|---|---|
| `action` | `click` (default), `type_text`, or a scroll/swipe action |
| `page_name` | Name of the page for the report (default: `step_name`) |
| `min_confidence` | Confidence threshold of step check 1 (default 0.35) |
| `optional` | Skip the step instead of failing if the target is not found |
| `skip_layout_check` | Do not require a page change after the step (step check 2) |
| `wait_before`, `wait_after` | Waiting time in seconds (defaults 0 and 1.5) |
| `repeat` / `count` / `times` | Repetitions of a scroll or swipe action |
| `clear`, `hide_keyboard` | Text input: clear the field first (default true), hide the keyboard afterwards |
| `is_password` | The field shows bullets, so the typed value is not checked against the field content (also set automatically if the step name or instruction contains e.g. "password" or "pin") |
| `skip_screenshot`, `skip_log` | Run the step without saving a screenshot or a log record (not used in the evaluated configs) |

3. Run and review:

```bash
python3 auto_screen.py test --app <app> --model Qwen_trained --mode dg
python3 auto_screen.py review --mode dg
```

---

## Optional features (added after the evaluation)

These features were added after the evaluation (branch `feat/improvements`, after tag `eval-final-2`). All of them are off by default. Without the options below, the comparison produces the same reports and diff images as `eval-final-2` (checked on the FF3 runs).

Four of them were evaluated afterwards with existing data only (threshold calibration, local assessment, sub-image refinement, approved baseline). Protocol, scripts and results are in `eval_ext/`. The protocol was committed before the runs. Stable screenshots, ignore regions, devices, dark mode, text size and right-to-left languages were not evaluated.

### Local VLM assessment of difference boxes

```bash
python3 compare_images.py --batch --app katapp --model Qwen_trained --version-baseline-root <previous screenshots> --ai-local
LOCAL_TRIAGE=1 ./run_pipeline_local.sh      # version comparison of the pipeline
```

Each box of the grayscale comparison is cut out of both screenshots and rated by a local VLM (default `Qwen/Qwen3-VL-4B-Instruct` from the local Hugging Face cache, override with `--ai-model` or `TRIAGE_MODEL_PATH`) as `defect`, `acceptable` or `unclear`. The largest boxes are rated first, at most `--ai-max-boxes` (default 8) per page. Screenshots do not leave the machine. The report shows the verdict per page and links an image with the rated boxes (red defect, orange unclear, gray acceptable, white not rated).

The verdicts only order the boxes for the reviewer. Evaluation (`eval_ext/`, E2) on all 296 grayscale boxes of the FF3 pages: the base model rated 43 of 44 boxes without a defect in the version comparison and 234 of 238 translation differences as acceptable, but only 5 of 14 boxes at seeded defects as defect (4 of the 10 detected defects). Skipping the boxes rated acceptable would lose 6 of 10 defects. Before relying on it, the model needs fine-tuning and an evaluation on separate data. `GATE_ON_AI_FAIL=1` makes the gate block runs with `fail` verdicts. It should not be used before such an evaluation.

### Approved baseline

```bash
python3 tools/approve_baseline.py --app katapp --source Qwen_trained/screenshots --label <commit> --note "checked by hand"
python3 tools/approve_baseline.py --app katapp --show
BASELINE_MODE=approved ./run_pipeline_local.sh
```

The version comparison then uses `baselines/<app>/screenshots` instead of the previous build, until the next approval. `baseline.json` records who approved which build and the SHA-256 of every screenshot. In CI, run the workflow **Approve visual baseline** with the ID of a successful CI run and set the repository variable `BASELINE_MODE=approved`. The approved artifact expires after 90 days.

### Stable screenshots and ignore regions

Config keys (top level of `configs/<app>.json`):

```json
"stability": { "hide_keyboard": true, "wait_for_idle": { "timeout": 3.0, "interval": 0.4, "max_change": 0.001 } },
"compare": {
  "mask_top": 140,
  "mask_bottom": 70,
  "ignore_regions": [ { "rect": [0, 400, 1206, 900], "pages": ["home_tab"], "folders": ["baseline_DE_DE"], "reason": "feed with changing content" } ]
}
```

- `hide_keyboard` hides the keyboard before the screenshot of every step except `type_text`. `wait_for_idle` waits until two consecutive frames differ by at most `max_change` of the pixels below the status bar. A step can turn both off with `"stabilize": false`.
- `ignore_regions` are excluded from both comparison methods and outlined in gray in the diff images. Without `pages` or `folders`, a region applies to all pages or folders. `--mask-top` and `--mask-bottom` override the masks, e.g. for other devices.

### Devices, dark mode, text size, right-to-left languages

```bash
python3 tools/run_matrix.py --app katapp --devices "iPhone 18 Pro" "iPhone 17e" \
    --appearance light dark --content-size large accessibility-large --dry-run
```

Each combination writes its screenshots to `<model>/screenshots_<variant>/`. Compare a variant only with the same variant of another build (`--version-baseline-root` / `--version-actual-root`) and set the masks for the device. The engine also reads `AUTOSCREEN_DEVICE_NAME`, `AUTOSCREEN_PLATFORM_VERSION`, `AUTOSCREEN_DEVICE_UDID`, `AUTOSCREEN_APPEARANCE`, `AUTOSCREEN_CONTENT_SIZE` and `AUTOSCREEN_VARIANT`, or `appearance` and `content_size` in `app_config`.

- **Real devices**: `app_config.extra_capabilities` is passed to Appium (e.g. `udid`, `xcodeOrgId`, `xcodeSigningId`, `updatedWDABundleId`). Appearance and text size can only be set on simulators. Not tested on a real device.
- **Right-to-left languages**: a flow with `"launch_language": "ar"` and optionally `"launch_locale": "ar_SA"` restarts the app with `-AppleLanguages`/`-AppleLocale` before the flow. The app must support the language. Mirrored layouts are only comparable with the same language of another build.
- The SoM box filter (status bar 110 px) and the heuristic fallbacks are tuned to the iPhone 18 Pro.

### Threshold calibration

```bash
python3 tools/calibrate_thresholds.py \
    --set <baseline root> <actual root> <annotations.json> --set <baseline root> <control root> none \
    --pages <page names> --out diff_results/calibration
```

Runs the unchanged comparison for a grid of thresholds and minimum box sizes and reports per method the found annotated defects (same rule as `ff3/ff3_eval.py`), box precision and boxes on pages without a defect. With τ = 38 and 14 px, it reproduces the FF3 results. Values calibrated on a set must be evaluated on other pairs.

### Sub-image refinement for Direct Grounding (Ferret-UI style)

`DG_SUBIMAGE_REFINE=1` (fine-tuned local model only): after the first prediction, the upper or lower half of the screenshot that contains it (10 % overlap) is passed again at almost twice the effective resolution, and the refined point is mapped back. If it lies more than 15 % of the screen height away (`DG_SUBIMAGE_MAX_SHIFT`), the first prediction is kept. Each step needs a second inference. The decision in the log records both points (`subimage_refinement`).

Evaluation (`eval_ext/`, E3): on 500 cases of the cleaned iOS-1K split, the refinement did not change the hit rate (68.2 % without, 67.8 % with, hit = at most 40 px from the element centre, McNemar p = 0.85, icons 51.6 % in both). It doubles the inference time per step.

---

## Known limitations

- **DG with cloud models at reduced resolution**: the coordinates of the cloud models are not converted back to the original screenshot. With `MAX_IMAGE_EDGE=1600`, Luna and Sol answered partly in the grid of the original and partly in the grid of the reduced image. Use the original resolution (`MAX_IMAGE_EDGE=0`). SoM is not affected.
- **Training prompt**: the prompt of the training data differs from the prompt at runtime. No validation split was used.
- **Thresholds**: not calibrated (see above).
- **Login page**: the keyboard is visible in some runs and not in others, which produces differences in the version comparison (optional: `stability.hide_keyboard`, not evaluated).
- **One device**: evaluated only on one simulator (iPhone 18 Pro, iOS 27.0) with two apps (KatApp, Bluesky). `tools/run_matrix.py` runs other variants but was not evaluated.
- **`auto_screen.py compare`**: forwards `--baseline` and `--actual`, which `compare_images.py` rejects as ambiguous. Call `compare_images.py` directly (step 4).
- **`auto_screen.py dataset`**: the arguments are not passed on, because `tools/prepare_dataset.py` uses fixed paths.
- **`requirements.txt`**: does not contain `gradio`.
