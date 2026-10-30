# Extended Evaluation Protocol (E1–E16)

Supplemental evaluation runs (base commit `2f50706`, evaluated baseline `eval-final-2` / `2d2693e`).

## E1: Threshold Sensitivity
- **Data**: FF3 (`B5` vs. `A1` with `ff3/seeded_defects.json`, control `A2` vs. `A1`), folders `DE`, `EN`, `ZH_CN`, `ZH_TW`.
- **Grid**: $\tau \in \{20, 26, 30, 34, 38, 42, 46, 50\}$, minimum box size $\in \{10, 12, 14, 16, 20\}$ px, masks 140/70 px.
- **Command**: `python3 tools/calibrate_thresholds.py --set ff3/runs/A1/screenshots ff3/runs/B5/screenshots ff3/seeded_defects.json --set ff3/runs/A1/screenshots ff3/runs/A2/screenshots none --pages 06_vehicle_tab 07_settings_tab 01_home_tab language_button login_field 13_language_choice --out eval_ext/results/e1`

## E2: Local VLM Diff Triage
- **Model**: `Qwen/Qwen3-VL-4B-Instruct` (base model, local), 24 px crop padding, $\tau = 38$, min box 14 px.
- **Data**: Version comparison `B5` vs. `A1` (58 boxes) and language comparison in `A1` (`DE` vs. `EN`, `ZH_CN`, `ZH_TW`, 238 boxes).
- **Result**: 296 boxes rated (`results/e2/`). 43/44 non-defect version boxes and 234/238 language boxes rated `acceptable`; 5/14 seeded-defect boxes rated `defect` (4/10 detected defects).
- **Command**: `python3 eval_ext/e2_triage.py --out eval_ext/results/e2`

## E3 & E3b & E3c: Sub-Image Refinement & 40 px Tolerance (Direct Grounding)
- **E3 (iOS-1K)**: 500 sampled cases (seed 42) from the cleaned iOS-1K evaluation split (`828x1792`), `Qwen3-VL-4B-V2`. Hit rate at $\le 40$ px: 68.2% without refinement vs. 67.8% with refinement ($p = 0.85$). Bounding-box check (`eval_ext/e3_boxcheck.py`, 432 matched boxes): 72.2% vs. 73.4% ($p = 0.30$).
- **E3b (App Flows)**: Applied `_refine_in_subimage()` to all 598 confirmed DG steps (`KatApp` 262, `Bluesky` 336). 590/598 refinements accepted; 27 shifted $> 40$ px within the same tappable menu row in Bluesky; 598/598 remained valid (`eval_ext/e3b_dg_flows.py`).
- **E3c (Tolerance Sensitivity)**: Across 4,350 DG records in `finetune_dataset_DG.jsonl`, 0 entries used `manual_click` or had distance $> 0$. On E3 predictions across tolerances 20–100 px, hit rate without refinement ranges from 62.8% to 74.2%, with 93.1% box agreement at 40 px (`eval_ext/e3c_toleranz.py`).

## E4: Approved Baseline Replay
- **Data**: Sequence `A1 -> B5 -> A2 -> B5` comparing rolling baseline vs. pinned baseline `A1`.
- **Command**: `python3 eval_ext/e4_baseline_replay.py --out eval_ext/results/e4`

## E5: Follow-Up Checks on App Flows and Prompts
- **E5 / E5b / E5c**: Unified English instruction for `triage_option` in `configs/katapp.json` and `cv2.RETR_CCOMP` dialog contour fix in `Qwen_trained/visual_agent.py`. KatApp DG completed 54/54 steps (4/4 flows); Bluesky DG completed 72/72 steps (3/3 flows); KatApp SoM completed DE and ZH-TW after dialog contour fix (`results/e5/`).
- **E5d (SoM Replay)**: Re-ran `ask_ai_for_element_id()` with `Qwen3-VL-4B-V2` on the 598 marked images from 2026-09-28 (`eval_ext/e5d_som_replay.py`). 598/598 matched both the original run and ground truth (median 2.0 s).
- **E5e (Base Model on Same Marked Images)**: `Qwen/Qwen3-VL-4B-Instruct` on the same 598 marked images matched 567/598 boxes vs. 598/598 for fine-tuned ($p = 9.3 \times 10^{-10}$, `eval_ext/e5e_compare.py`). 16/31 differences targeted the same UI control; 15 targeted a different element (`triage_dropdown` ZH-CN/ZH-TW, `triage_option` DE).
- **E5f (Login Step Check)**: Base model with the current prompt selects ID 5 on both the 2026-10-13 and 2026-09-28 `initial_login_save_button` images (`results/e5f.json`).
- **E5g (DG Image Size Check)**: Re-running the first 6 steps of `Qwen_untrain` DG from 2026-10-24 confirms 6/6 exact output match with `max_edge=1280` (`eval_ext/e5g_dg_imagesize.py`).
- **E5h (SoM Box vs. ID Permutation)**: Tested original numbering (A0), label swap (A), 20 random permutations (B), and merged label+dropdown box (C) on `triage_dropdown` (`eval_ext/e5h_renumber.py`). Dropdown chosen in 17/20 (EN), 18/20 (DE), 18/20 (ZH-CN), and 15/20 (ZH-TW) permutations.
- **E5i (Cloud Models on Versioned SoM Markings)**: GPT-6 Sol, GPT-6 Luna, and Claude Sonnet 5.5 at `MAX_IMAGE_EDGE=0` and `1600` selected the dropdown (Box 8) in all 24/24 calls (`eval_ext/e5i_cloud_som.py`).

## E6: Native Resolution vs. Downscaled Input (Local Fine-Tuned Model)
- **E6-SoM**: `LOCAL_MAX_IMAGE_EDGE=0` on the 598 SoM steps from E5d (`results/e6_som_original.jsonl`): 595/598 matched vs. 598/598 at 1600 px ($p = 0.25$), median 6.3 s vs. 2.0 s.
- **E6-DG**: Paired comparison of 1280 px vs. native resolution on the 598 DG steps (`eval_ext/e6_dg_fullsize.py`, `results/e6_dg.jsonl`): 598/598 exact reproduction at 1280 px (median 9.0 s); 569/598 within 40 px at native resolution plus 29/29 valid taps within the same tappable row in Bluesky (598/598 valid, median 20.4 s).

## E7: Target Box Provenance in iOS-1K
- **Check**: Matches target centers (`gt_x`, `gt_y`) within 3 px against raw MUIAnno annotations (`eval_ext/e7_boxherkunft.py`, `results/e7_boxherkunft.json`).
- **Result**: Evaluation split: 1,877 / 2,008 instructions (93.5%, 231 images). Training split: 3,497 / 7,779 examples (45.0%, 737 images).

## E8: Instructions Without Position Hints
- **Setup**: Removed positional hints from 21 instructions across 398 of the 598 SoM steps (`eval_ext/e8_lagehinweise.py`, `results/e8_summary.json`).
- **Result**: 383/398 (96.2%) matched vs. 398/398 with hints ($p < 0.001$). All 15 misses occurred on a single icon-only instruction in Bluesky (`go to # page.`).

## E9: Base Model Direct Grounding on iOS-1K
- **Setup**: `Qwen/Qwen3-VL-4B-Instruct` on the same 500 iOS-1K cases as E3 (`eval_ext/e9_dg_base_ios1k.py`, `results/e9_summary.json`).
- **Result**: Hit rate at $\le 40$ px: base model 331/500 (66.2%) vs. fine-tuned 341/500 (68.2%, $p = 0.24$). Base model produced 33/500 unparsed outputs (fallback) vs. 0/500 for fine-tuned.

## E10: Cross-Language Comparison Ranking on FF3 Runs
- **Setup**: Evaluates whether seeded-defect pages in `B5` rank higher by EN-vs-DE diff box count than in `A1` and `A2` (`eval_ext/e10_sprachvergleich.py`, `results/e10_sprachvergleich.json`).
- **Result**: Precision@6 is 0.667 across `A1`, `A2`, and `B5` (AUC 0.694–0.722); natural translation length differences dominate the cross-language box ranking.

## E11: Blind Second-Rater Verification
- **Setup**: 80 shuffled cases (40 fine-tuned DG steps from Part A, 31 base-model SoM deviations + 9 controls from Part B) rated blind (`eval_ext/e11_zweitbewertung.py`, `results/e11_summary.json`).
- **Result**: Part A: 40/40 rated correct (95% Wilson CI: 91.2–100%). Part B: 35/36 non-unclear cases agreed with primary rater (97.2%, Cohen's $\kappa = 0.93$; on the 31 deviations: 26/27, 96.3%, $\kappa = 0.92$).

## E12: Target Element Size Distribution
- **Setup**: Bounding box dimensions across the 598 verified SoM steps (`eval_ext/e12_zielgroesse.py`, `results/e12_zielgroesse.json`).
- **Result**: Shorter side minimum 26 px (8.7 pt), 5th percentile 39 px, median 64 px (21.3 pt); 79.9% have shorter side $< 80$ px.

## E13: Icon Miss Classification on iOS-1K
- **Setup**: Categorized all 95 icon misses of the fine-tuned model in `results/lora.jsonl` (`eval_ext/e13_symbole.py`, `results/e13_summary.json`).
- **Result**: Wrong dataset reference box in 61/95 cases (32 model also wrong, 21 model actually correct, 8 fallback with wrong reference); 16 ambiguous instructions (`Tap the action icon`), 3 same UI control, 2 similar icon, 10 other.

## E14: Semi-Automatic Step Verification of Cloud Runs
- **Setup**: Checked all 16 primary cloud runs (4,747 steps) against verified SoM boxes and confirmed DG points (40 px tolerance), with manual review of the remaining 284 steps in 88 groups (`eval_ext/e14_cloud_schritte.py`, `results/e14_summary.json`).
- **Result**: 4,744 / 4,745 evaluated steps correct (2 valid rejections on unboxed/unreached targets). Single miss: Claude Sonnet 5.5 SoM (1600 px) on Bluesky `check_following_page` (EN).

## E15: Third App Generalization (Simplenote iOS)
- **Setup**: Native iOS app (`configs/simplenote.json`, 7 setup steps + 4 steps across EN, DE, ZH, 5 loops per mode) with `Qwen_trained` (`eval_ext/e15_simplenote_review.py`, `results/e15_summary.json`).
- **Result**: DG completed 15/15 flows (67/67 steps correct). SoM completed 0/15 flows because `RETR_EXTERNAL` merged the login email field, password field, and submit button into a single contour box on the password screen.

## E16: Full Matrix Rerun on Versioned Codebase
- **Setup**: 6 runs (5 loops each) on `HEAD` (`83ddc34` + `RETR_CCOMP` dialog fix and unified `triage_option` instruction) in `Qwen_trained_e16` and `Qwen_untrain_e16` (`eval_ext/e16_laeufe.sh`, `eval_ext/e16_schritte.py`, `results/e16_summary.json`).
- **Result**:
  - Fine-tuned DG: KatApp 20/20 flows (262/262 steps), Bluesky 15/15 flows (336/336 steps).
  - Fine-tuned SoM: KatApp 16/20 flows (232/240 steps), Bluesky 14/15 flows (316/317 steps).
  - Base SoM: KatApp 0/20 flows (`initial_login_save_button` miss, 1/2 evaluated steps), Bluesky 10/15 flows (272/275 steps).
