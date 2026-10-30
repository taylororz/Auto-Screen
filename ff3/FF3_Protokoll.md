# FF3 Protocol: Version Comparison with Seeded Visual Defects

Date: 25.10.2026 (updated after evaluating B1 vs. A1). This protocol documents the decisions, commits, and deviations prior to runs A1, A2, and B1.

## 1. Versions

| Component | Reference | Notes |
|---|---|---|
| Auto-Screen (evaluation code) | Tag `eval-final` → `b119068` | The evaluation uses `compare_images.py` at this commit (blob `8d03680373e6`). Not pushed. Addendum 26.10.2026: In the working copy, only the text of the HTML report was changed. When the folder name is the same on both sides, the report is now titled "Version comparison". Comparison logic, thresholds, and numbers are unchanged (verified: 58 and 53 boxes before and after the change). The tag `eval-final` still points to `b119068`. In addition, the `--version-baseline-root` option was added. It compares each language against the folder of the same name from the previous release and generates a separate report. The pipeline thus creates two reports: the language comparison (`diff_results/`) and, when `PREVIOUS_SCREENSHOTS` is set, the version comparison (`diff_results/version_comparison/`). Verified with A1 vs. B5: DE 58 and 53 boxes, EN and ZH 0. |
| KatApp Build A | `fdb6f79` (Branch `feat/zh-localization`) | Original version + Chinese translation (zh_CN, zh_TW). |
| KatApp Build B | `af44aec` (Branch `seeded-defects`), built as **Build B5** | Build A + 14 commits D01–D16 (excluding D07, D12). Rebuilt from a clean worktree; the German strings of D11 and D14 are lengthened (committed `de.json`). Build B1 is superseded, see Section 12. |

Uncommitted, but identical in both builds (see Section 4):
`flavors.dart` (API address `127.0.0.1:9997`), iOS deployment target, Podfile.

## 2. Decisions

1. **Eval tag**: `eval-final` was set to `b119068`. The directory `~/Masterarbeit/Auto-Screen` is not a Git repository; the experiments use `~/Auto-Screen`.
2. **Build A**: only the Chinese translation was committed (excluded: dev API URL, location key migration, casualty strings, iOS build files).
3. **Number of defects: 14 instead of 20.** Rationale: Only strings that do not belong to navigation and whose layout remains unchanged in EN/ZH can be targeted. With the constraint "at most 3 defects per page", the upper bound is 14. The distribution is: Truncated Text 4 (D01–D04), Clipped Text 3 (D05, D06, D08), Overlap 3 (D09–D11), Displaced Element 4 (D13–D16).
4. **Omitted**: D07 (login token label, `labelText` cannot be truncated or clipped) and D12 (login title overlap; only one text element available on the page). The IDs are retained for traceability.
5. **Partial commits**: `triageOrtSelection.dart` contains the location key migration. Commits D11 and D14 contain only the respective text changes.
6. **Page mapping**: Login page: `login_screen.dart`; Home screen: `home_screen.dart` and `triageOrtSelection.dart` (page `triage_location_menu`); Language dialog: `setting_screen.dart`. The page mapping for `home_slogan` (D16) is not confirmed by a screenshot.
7. **Unused pages** (control without defects): `scan_page`, `activation_token_page`, `language_choice_page`.

## 3. Defect Table (Build B, `351e8cb`)

| ID | Class | Page | ARB Key (de.json) | Commit |
|---|---|---|---|---|
| D01 | Truncated Text | overview_page | triages_overview_screen_title | 34cfe35 |
| D02 | Truncated Text | vehicle_page | vehicle_screen_last_sent | 49d72ab |
| D03 | Truncated Text | settings_page | setting_screen_data_privacy | 16b02dc |
| D04 | Truncated Text | login_page | login_screen_title | d4315b2 |
| D05 | Clipped Text | overview_page | no_upload_queued | 35042a2 |
| D06 | Clipped Text | vehicle_page | vehicle_screen_description | cc84cd3 (vehicle tracking description, third line clipped) |
| D08 | Clipped Text | language_dialog | choose_language_home_screen | 9e4f6ae |
| D09 | Overlap | vehicle_page | vehicle_screen_tracking_active | 7a4b2d8 (only visible when tracking is active, see Section 10) |
| D10 | Overlap | settings_page | setting_screen_access_data | 953bd8a |
| D11 | Overlap | triage_location_menu | home_select_current_triage_location | 066f5d1 |
| D13 | Displaced Element | overview_page | triages_overview_screen_catastrophe | 7058b23 |
| D14 | Displaced Element | triage_location_menu | home_current_triage_location | 351e8cb |
| D15 | Displaced Element | settings_page | settings_screen_appearance_mode | f705ab8 |
| D16 | Displaced Element | triage_location_menu | home_slogan | 8cfd9a2 |

All modified German strings are in `de.json` only. EN and ZH files were not modified for these defects.

## 4. Build Environment (identical for A and B)

- Xcode 27. The deployment target was raised from 13.0 to 15.0 (Podfile, `post_install`, `Runner.xcodeproj`) because Xcode 27 requires simulator builds to target at least 15.0. This change is uncommitted.
- `flutter config --no-enable-swift-package-manager` (global Flutter setting, CocoaPods instead of SPM).
- `flutter build ios` fails on the generic target "Any iOS Simulator Device" (exit 255). Therefore, the build is executed via `xcodebuild -scheme dev -configuration Debug-dev -destination 'platform=iOS Simulator,name=iPhone 18 Pro'`, and the output is copied to `build/ios/iphonesimulator/Runner.app`. Bundle ID: `de.kfru.katapp.dev` (matches `katapp.json`).
- Build B was built successfully. Build A was built subsequently using the same setup.

## 5. Checks and Results

| Check | Status |
|---|---|
| `flutter analyze` (modified files) | No errors; three existing hints |
| Build A (`fdb6f79` + same configuration, without defects) | Successful (`xcodebuild dev`) |
| Build B2 (`af44aec`, defects active only in German) | Successful (`xcodebuild dev`) |
| Runs A1, A2, B1 | Complete (4 languages, 0 errors, training data restored after each run) |
| EN_US, ZH_CN, ZH_TW: B1 vs. A1 (excluding login screenshots) | Identical: 33/33, 107/107, 33/33 pages, difference 0 |
| Noise control A1 vs. A2 (EN/ZH) | Difference 0 |
| DE: B1 vs. A1 (excluding login screenshots) | 4 of 36 pages identical; maximum 4.2% pixels |
| Yellow-black stripes (overflow warning) | None; yellow areas (video page, Bluesky menu) are identical in A1 and B1 |
| Navigation | Executed consistently across all runs |

### Excluded Screenshots (Decision)

The following three screenshots are excluded from all comparisons:
`login_page.png`, `login_save_button.png`, `10_login_save_button.png` (in all language folders).

Reason: In the screenshot taken after entering the token, the iOS keyboard was visible in B1 but not in A1. This is a timing artifact of the run rather than a property of the defect. Without this exclusion, these pages would falsely appear as differences in EN, ZH-CN, and ZH-TW.

Consequence for evaluation: D04 (login page title) is located in the upper area of the screen and is unaffected by the exclusion because the title and keyboard are separate. The evaluation of the login page for D04 is checked on `09_login_field.png` (Section 15).

## 6. Deviations from the Experiment Plan

- 14 instead of 20 defects; rationale in Section 2.
- Overlap class: Defects D09–D11 produce overlaps via text wrapping (fixed height, `overflow: visible`) rather than overlapping an icon.
- Displaced Element class: D13–D16 use fixed widths with a horizontal, non-scrollable `ScrollView`. The text is cut off rather than pushed aside, serving as an approximation of the concept.
- The sample size is small (14); per-class recall figures are based on 3–4 defects and have limited statistical power.

## 7. Steps Completed

1. Build A from `fdb6f79` with the same configuration changes (Podfile, deployment target, `flavors.dart`), without any defect commits.
2. Capture screenshots of A and B in EN, ZH-CN, ZH-TW, and DE across the 10 pages.
3. Verify: EN/ZH in B identical to A; DE defects visible; no overflow stripes; navigation works.
4. Execute runs A1, A2 (A) and B1 (B) with Auto-Screen; populate `seeded_defects.json` with ground-truth bounding boxes (`bbox_picker.py`); run `ff3_eval.py`.

## 8. Runs

| Run | Build | Result | Folder |
|---|---|---|---|
| A1 | A (`fdb6f79`) | Complete, 223 files | `runs/A1/` |
| A2 | A (`fdb6f79`) | Complete | `runs/A2/` |
| B1 | B2 (`af44aec`) | Complete | `runs/B1/` |
| B1_v1_ungated | B (`351e8cb`) | Superseded: EN/ZH not identical, D11/D14 missing in build | `runs/B1_v1_ungated/` |

Run B1_v1 is not part of the evaluation and is documented only for traceability.

Verification: `ff3/results/verification_pixel_diffs.csv` (all pairs, including login screenshots; the exclusions above apply during evaluation).

## 10. Manual Annotation (Ground Truth)

The bounding boxes were drawn manually using `bbox_picker.py` on the B1 screenshot with A1 shown side by side as reference. The boxes are logged in `ff3/bbox_log/bboxes_settings.txt`, and the ground truth is stored in `ff3/seeded_defects.json`.

The mapping "filename → page" follows the visual content of the file rather than its filename. Several filenames show the same screen (identical MD5) or a different screen than their name suggests. Examples: `overview_page.png` shows the scan screen, `settings_page.png` shows the home screen, `04_scan_tab.png` shows the home screen. Evaluation uses the filename under which the defect is actually visible.

| Defect | File (Screen) | Box `[x, y, w, h]` | Status |
|---|---|---|---|
| D01 | `06_vehicle_tab.png` (Overview) | [276, 198, 643, 145] | annotated |
| D03 | `language_button.png` (Settings) | [326, 1861, 579, 125] | annotated |
| D04 | `09_login_field.png` (Login) | [320, 198, 527, 125] | annotated |
| D06 | `07_settings_tab.png` (Vehicle tracking) | [61, 439, 1092, 168] | annotated |
| D08 | `13_language_choice.png` (Language dialog) | [171, 958, 739, 166] | annotated |
| D10 | `language_button.png` (Settings) | [78, 1293, 792, 125] | annotated |
| D15 | `language_button.png` (Settings) | [227, 751, 748, 148] | annotated |
| D16 | `01_home_tab.png` (Home) | [34, 218, 1153, 201] | annotated |
| D02 | `07_settings_tab.png` | empty | not identified |
| D05 | `06_vehicle_tab.png` | empty | not identified |
| D09 | `07_settings_tab.png` | empty | not identified |
| D11 | `01_home_tab.png` (Title "Wähle den Triage-Ort") | [96, 858, 1014, 99] | from element position (Appium page source, ×3), defect barely perceptible visually |
| D13 | `06_vehicle_tab.png` | empty | not identified |
| D14 | `01_home_tab.png` (Label "Aktuell ausgewählt:") | [96, 1014, 405, 60] | from element position (Appium page source, ×3), defect barely perceptible visually |

**Correction:** The box initially recorded as D09 belongs to D06. The text clipping (third line clipped) matches the `ClipRect` height 40 of D06. D09 is a string for active tracking and was not visible during the run.

**Not identified (empty GT):** Defects D02, D05, D09, D11, D13, and D14 were not visible in the screenshot or not manually identifiable in the B1 run. D02, D05, D09, and D13 require an application state that was not reached during the run (transmitted location, active tracking, queued uploads, disaster mode). D11 and D14 are not visually discernible on the home screen screenshot in B1. An empty GT counts as not detected in the evaluation (see Section 11).

**Addendum D11, D14 (added after initial evaluation):** The element frames were obtained from the Appium page source of the installed Build B (`builds/B2`). The ARB key mapping is: D11 = `home_select_current_triage_location` (title), D14 = `home_current_triage_location` (label). The box previously used for D14 ("Schadensort") was incorrect and was discarded.

**Build finding:** In Build B1, the German strings for both keys were **not lengthened**. The working copy of `de.json` contained the original strings from Build A (`Aktuell ausgewählt:`, `Wähle den Triage-Ort`), whereas the committed version in `af44aec` contains the lengthened strings. Thus, the tested Build B1 did not fully match `af44aec`. While the defect layouts D11 and D14 were present in code, they produced no visible effect in B1 (commits `066f5d1` and `351e8cb` had no effect in Build B1).

**Pixel check inside the boxes:** The crops in `ff3/gt_check/D11.png` and `D14.png` show identical text for D14, shifted by only a few pixels. The pixel difference within the box is 6.45% (D11) and 22.77% (D14), caused by the layout shift described in Section 13 rather than the defect itself.

## 11. Sample Decision

Proposals to remove the six non-visible defects from the evaluation or replace them with new ones were rejected. Removing them would restrict recall to easily visible cases, and replacing them after inspecting results would constitute post-hoc selection. All 14 defects remain the base sample. Reported metrics:
- Recall across all 14 defects,
- Recall across the 8 defects with a manual bounding box (noted as the subset of visually observable defects).

Any supplementary experiment with new defects would require a pre-specified protocol and explicit labeling as a post-hoc addition.

## 12. Evaluation B1 vs. A1 (DE)

**Version 2 (after adding D11 and D14 boxes):** The result file `summary.json` is the updated version; the previous version is preserved as `summary_v1.json` and `page_results_v1.csv`.

| Method | Recall | Box TP | Box FP | Precision | Missed |
|---|---|---|---|---|---|
| Pixel | 10/14 (71.4%) | 15 | 38 | 28.3% | D02, D05, D09, D13 |
| Grayscale (cv) | 10/14 (71.4%) | 14 | 44 | 24.1% | D02, D05, D09, D13 |

**Important:** In B1, the two additional hits D11 and D14 are not true defect detections: boxes `[94, 1020, 1018, 99]` and `[95, 876, 340, 61]` lie on the layout shift while the text in B1 was unchanged. Consequently, Build B5 was built from a clean worktree (Section 16).

**Version 1 (before adding D11/D14 boxes):** Pixel 8/14 (57.1%), Grayscale 8/14 (57.1%), Precision 22.6% and 19.0%.

Command (pages specified explicitly because the default list in `ff3_eval.py` does not match the filenames):
```bash
python3 ff3_eval.py --tool-dir ~/Auto-Screen --baseline runs/A1/screenshots --actual runs/B1/screenshots --gt seeded_defects.json --out results/B1_vs_A1 --pages 06_vehicle_tab 07_settings_tab 01_home_tab language_button login_field 13_language_choice
```
The control comparison A2 vs. A1 was executed with the same pages without `--gt`.

Table Version 1 (superseded, files `summary_v1.json` and `page_results_v1.csv`):

| Comparison | Method | Recall | Total Boxes | Box TP | Box FP | Precision | Boxes on Clean Pages |
|---|---|---|---|---|---|---|---|
| B1 vs. A1 | Pixel | 8/14 (57.1%) | 53 | 12 | 41 | 22.6% | 0 |
| B1 vs. A1 | Grayscale (cv) | 8/14 (57.1%) | 58 | 11 | 47 | 19.0% | 0 |
| A2 vs. A1 | Pixel / cv | – | 0 | – | – | – | 0 |

Recall by class (Pixel): Clipped Text 2/3, Truncated Text 3/4, Displaced Element 2/4, Overlap 1/3. Missed: D02, D05, D09, D11, D13, D14.

These figures cover 24 page comparisons (6 DE pages with defects, 4 language folders, 24 pairs in total). The defect-free pages are EN and ZH as well as the remaining DE pages.

## 13. Finding: Layout Shift in DE (Implementation Side Effect)

The difference images show boxes on elements without code changes, for example on the headings "Erscheinung", "System", and "Info" and on the buttons "Sprache", "Datenschutz", and "Benutzerhandbuch" (page `language_button`).

Measured via template matching between A1 and B1 (same page, DE):
- "Erscheinung": dy = −6 px
- "System": dy = −6 px
- "Sprache" button: dy = −6 px
- "Info": dy = +6 px

The shift is vertical with opposite signs in different sections, likely caused by a height change of a child widget inside a `Column` with `Spacer`.

The shift occurs only in DE because the defects are gated to DE. EN and ZH are identical (Section 5).

Consequences:
- A substantial portion of the false-positive boxes stems from this shift rather than detector noise alone.
- The "clean" regions on DE pages in Build B are affected by layout shifts, which must be discussed as a threat to validity in the thesis.

## 14. Attempted Layout Fix (B3, B4): Unsuccessful

The goal was to prevent the DE defects from altering the height of neighboring elements (Section 13).

| Version | Commit (katapp) | Helper | Run | Result |
|---|---|---|---|---|
| B1 / B2 | `af44aec` | Ternary expression: DE renders defect, otherwise original | `runs/B1` | Vertical shift ±6 px (Settings), EN/ZH identical to A1 |
| B3 | `f611076` | Stack: original with `Visibility(maintainSize)` as layout box, defect in `OverflowBox` | `runs/B3_fix` | EN/ZH identical; DE shift increased to ±36 px |
| B4 | `dd4b6fe` | Stack: original with `Opacity(0)` + `IgnorePointer` + `ExcludeSemantics` | `runs/B4_fix` | Screenshots identical to B3 (same MD5); same shift |

Finding: The placeholder mechanism (`Visibility` vs. `Opacity`) is not the cause, as B3 and B4 produce identical screenshots. The larger shift is introduced by the `Stack` structure itself.

Decision: The layout fix is not pursued further. Commits `f611076` and `dd4b6fe` were removed from the `seeded-defects` branch via `git reset --keep af44aec` (HEAD = `af44aec`).

## 15. Next Steps

1. Report the evaluation results alongside the layout shift analysis from Section 13.
2. D04 (login title) is included in the DE evaluation using `09_login_field.png` (where the keyboard is not shown). In EN and ZH, `login_field` produces zero boxes.
3. Build B is `af44aec`.

## 16. Rebuild B5 and Evaluation B5 vs. A1

**Motivation:** In Build B1, the German strings of D11 and D14 were not lengthened because of an uncommitted working copy of `de.json` (Section 12).

**Procedure:** Built from a clean worktree at `af44aec` with the required build configuration changes (excluding the stale `de.json`), followed by `pod install` and `flutter pub get`. After fixing `Generated.xcconfig` and `flutter_export_environment.sh` to point to the worktree, the app bundle contained the lengthened strings.

**Verification:** The B5 bundle contains `Aktuell ausgewählter Triage-Ort der Einsatzstelle:` and `Wähle den Triage-Ort der aktuellen Einsatzstelle`. In the B5 screenshot, D14 is visible as clipped text, and D11 shows "Wähle den Triage-Ort der" before being cut off.

**Run:** B5 (`runs/B5`), training data restored (`RESTORE_OK`).

**Evaluation** (same command as Section 12, `--actual runs/B5/screenshots`, output `results/B5_vs_A1`):

| Method | Recall | Missed | Box TP | Box FP | Precision |
|---|---|---|---|---|---|
| Pixel | 10/14 (71.4%) | D02, D05, D09, D13 | 15 | 38 | 28.3% |
| Grayscale (cv) | 10/14 (71.4%) | D02, D05, D09, D13 | 14 | 44 | 24.1% |

Recall by class (Pixel): Clipped Text 2/3, Truncated Text 3/4, Displaced Element 3/4, Overlap 2/3.

**Limitation:** On the home screen, the boxes for D11 and D14 partially overlap with the layout-shift boxes from Section 13. Whether detection is triggered solely by the defect or also by the layout shift cannot be completely separated at the bounding-box level. The results of B5 are therefore reported with this caveat.

**EN, ZH:** B5 vs. A1 are identical (EN 31 identical, ZH_CN 103, ZH_TW 32; 0 differences).

The results of B1 (Section 12) are retained for documentation, while B5 serves as the evaluated Build B.
