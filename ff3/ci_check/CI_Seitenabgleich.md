# FF3: Comparison of CI Pages with FF3 Pages

Date: 26.10.2026. This check examines the pages that appear only in the version comparison of the CI pipeline and are not among the 24 page comparisons of FF3: `triage_option_page`, `scan_page`, and `activation_token_page`. In addition, `language_choice_page` was checked because it is listed as a control page in Section 2, Point 7.

## Data Basis

- CI runs (GitHub Actions, version comparison): logs `109371125289` (Run 1), `109376077461` (Run 2), `109394004200` (Run 3), `109398768732` (Run 4). Runs 1 and 3 compare `7b66b5664c31` → `0576685c0ecf`; Runs 2 and 4 compare the reverse direction.
- Locally available CI artifact: `visual-regression-report-2` from **Run 1** (current = `0576685c0ecf`, `run_id` 40353105556). The artifact from Run 4 is not available locally.
- Local FF3 runs: `runs/A1`, `runs/B5`, evaluation `results/B5_vs_A1`.
- Threshold for pixel comparison in this check: maximum channel difference > 38 (same as `threshold` in the CI report).

## Results

1. The box counts are identical across all four CI runs (`triage_option_page` 8/9, `scan_page` 11/12, `activation_token_page` 14/14, each cv/pixel).
2. The CI screenshots of `0576685c0ecf` are identical to A1 within the threshold for all four pages (0 differing pixels).
3. The difference CI vs. B5 is pixel-identical to the difference A1 vs. B5. Thus, the CI difference corresponds to the difference between A1 and B5.
4. In terms of content, the four pages show views that FF3 already evaluates:

| CI Page | Content | Identical to (A1 and B5) | CI cv/Pixel | FF3 B5_vs_A1 cv/Pixel | Cause of Boxes |
|---|---|---|---|---|---|
| `scan_page` | Home screen | `01_home_tab` | 11/12 | 11/12 | D11, D14, D16 and downstream shift (selection field, "Starte die Triage") |
| `activation_token_page` | Settings page, token dialog not open | `language_button` | 14/14 | 14/14 | D03, D10, D15 and layout shift from Section 13 |
| `language_choice_page` | Language dialog | `13_language_choice` | 20/14 | 20/14 | D08 and further differences (see FF3 evaluation) |
| `triage_option_page` | Home screen with open dropdown menu | `03_triage_option` (not among the 24 comparisons) | 8/9 | – | D11, D14, D16 and downstream shift. No boxes inside the menu |

## Conclusions

- The differences on these pages are due neither to different screen content nor to a new shift. They correspond to defects D03, D08, D10, D11, D14, D15, and D16 as well as the layout shift from Section 13.
- Section 2, Point 7 must be corrected: `scan_page`, `activation_token_page`, and `language_choice_page` are not defect-free control pages.
- The CI pages must not be added to the 24 page comparisons; otherwise defects would be counted twice.
- Open item: The direction of the labels in `summary_report.json` (`baseline_label`, `actual_label`) should be confirmed on the artifact from Run 4.

## Files

- `abgleich.csv`: identity check and pixel counts per page
- `bilder/vergleich_<seite>.png`: A1 | B5 | CI (Run 1) side by side
- `bilder/ci_diff_<seite>.png`, `bilder/ci_diff_<seite>_pixel_matching.png`: difference images from the CI report (Run 1, scaled down)

![scan_page](bilder/vergleich_scan_page.png)
![activation_token_page](bilder/vergleich_activation_token_page.png)
![triage_option_page](bilder/vergleich_triage_option_page.png)
![language_choice_page](bilder/vergleich_language_choice_page.png)
