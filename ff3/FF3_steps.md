# FF3 Seeded-Defect Experiment: Procedure

Objective: Measure the ability of Auto-Screen's two comparison methods (pixel comparison / grayscale comparison) to detect localization display defects in **version comparison**, evaluated via Recall and Precision (corresponding to Research Question FF3).

---

## 0. Experimental Design

| Item | Configuration |
|---|---|
| System under Test | KatApp (Flutter), iPhone 18 Pro Simulator, iOS 27 |
| Build A | Original version (with Simplified and Traditional Chinese translations added) |
| Build B | Build A with 14 seeded defects: Truncated Text (4), Clipped Text (3), Overlap (3), Displaced Element (4) (D07 and D12 omitted, see `FF3_Protokoll.md`) |
| Defect Scope | **German locale only** (`baseline_DE_DE`). EN, ZH-CN, and ZH-TW pages serve as defect-free controls |
| Pages | 10 pages from `katapp.json` × 4 languages = 40 page comparisons |
| Runs | A1, A2 (two runs of Build A), B1 / B5 (Build B) |
| Comparisons | B vs. A1 → Recall, Precision. A2 vs. A1 → false positives from rendering noise |
| Detection Criterion | At least one detected difference box intersects the annotated ground-truth defect bounding box |

Defect classes match Table 2.1 of the thesis: **Truncated Text / Clipped Text / Overlap / Displaced Element**.

---

## 1. Seeding Defects (Build B)

Principle: **Defects are triggered by longer German strings** combined with fixed layout constraints in code. These constraints must not alter the visual appearance in English or Chinese where strings are shorter, simulating rigid UI layouts that break under text expansion during localization.

Create a branch (`git checkout -b seeded-defects`) and commit each defect individually with its defect ID (`D01`…).

### Implementation of the Four Defect Classes (Flutter)

**A. Truncated Text (D01–D04): horizontal truncation with ellipsis**
- Wrap the text widget with `maxLines: 1, overflow: TextOverflow.ellipsis` inside a `SizedBox(width: …)` or `ConstrainedBox(maxWidth: …)` sized so that the English string fits.
- Lengthen the corresponding German string in `de.json` (e.g., using compound nouns).

**B. Clipped Text (D05, D06, D08): vertical clipping**
- Place the text inside a `SizedBox(height: <single-line height>)` and `ClipRect` with `maxLines: 2` (or unlimited) and lengthen the German string so it wraps onto a second line that gets clipped vertically without an ellipsis.

**C. Overlap (D09–D11): overlapping elements**
- Place elements at fixed positions in a `Stack` / `Positioned` or fixed-height container with `overflow: TextOverflow.visible` so that the longer German text overlaps adjacent UI elements.

**D. Displaced Element (D13–D16): element pushed outside the visible viewport**
- Place a row inside a `SingleChildScrollView(scrollDirection: Axis.horizontal, physics: NeverScrollableScrollPhysics())` so that longer German text pushes trailing elements outside the visible screen area.

### Important Precautions

1. **Do not trigger Flutter's yellow-black overflow warning stripes.** Simulator runs use debug builds, where a `Row`/`Column` `RenderFlex` overflow renders yellow-and-black diagonal stripes that any pixel diff would trivially flag. Verify every screenshot to ensure no overflow stripes appear.
2. **Keep control languages unchanged.** After building B, verify that EN, ZH-CN, and ZH-TW screens remain pixel-identical to Build A.
3. **Distribute the 14 defects across views** (at most 3 per view), keeping some views defect-free as controls.
4. **Keep defects realistic and visible** (e.g., clipped characters or partial overlap rather than removing entire sections).

---

## 2. Building and Running

```bash
# Build A
git checkout main && flutter build ios --simulator --debug
# Run Auto-Screen and copy the screenshots folder:
cp -R <Auto-Screen>/Qwen_trained/screenshots runs/A1/screenshots
# Second run of Build A for noise control:
cp -R <Auto-Screen>/Qwen_trained/screenshots runs/A2/screenshots

# Build B
git checkout seeded-defects && flutter build ios --simulator --debug
# Run Auto-Screen and copy screenshots:
cp -R <Auto-Screen>/Qwen_trained/screenshots runs/B5/screenshots
```

- All runs use the **same simulator, iOS version, and system settings** (appearance, text size).
- Any missing pages are reported by `ff3_eval.py`.

---

## 3. Annotating Ground-Truth Defect Boxes

1. Copy `seeded_defects_template.json` to `seeded_defects.json` and fill in the 14 entries.
2. Obtain each `bbox` with `bbox_picker.py` by dragging a rectangle around the defect region and pressing ENTER:

```bash
pip install opencv-python
python3 bbox_picker.py runs/B1/screenshots/baseline_DE_DE/language_button.png runs/A1/screenshots/baseline_DE_DE/language_button.png
```

Draw the bounding box tightly around the affected UI element.

---

## 4. Evaluation

```bash
# Defect detection
python3 ff3_eval.py --tool-dir ~/Auto-Screen \
  --baseline runs/A1/screenshots --actual runs/B5/screenshots \
  --gt seeded_defects.json --out results/B5_vs_A1

# Noise control (without --gt)
python3 ff3_eval.py --tool-dir ~/Auto-Screen \
  --baseline runs/A1/screenshots --actual runs/A2/screenshots \
  --out results/A2_vs_A1
```

`ff3_eval.py` calls `compare_images.py` unchanged and evaluates the exact boxes shown in the HTML report. Outputs:
- `summary.json`: overall and per-class recall, box-level precision, page-level statistics, and missed defect IDs
- `page_results.csv`: per-page breakdown
- `diffs/`: annotated difference images

---

## 5. Limitations

- 14 seeded defects tested on one application and one simulator device.
- Defect locations and types were selected by the author.
- Defects were seeded in the German locale only.
- Tested on simulator debug builds due to iOS Simulator constraints.

