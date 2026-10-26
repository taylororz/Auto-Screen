# FF3: Abgleich der CI-Seiten mit den FF3-Seiten

Stand: 26.10.2026. Geprüft wurden die Seiten, die nur im Versionsvergleich der CI-Pipeline vorkommen und nicht unter den 24 Seitenvergleichen von FF3 sind: `triage_option_page`, `scan_page` und `activation_token_page`. Zusätzlich wurde `language_choice_page` geprüft, weil sie in Abschnitt 2 Punkt 7 als Kontrollseite geführt wird.

## Datenbasis

- CI-Läufe (GitHub Actions, Versionsvergleich): Logs `109371125289` (Run 1), `109376077461` (Run 2), `109394004200` (Run 3), `109398768732` (Run 4). Runs 1 und 3 vergleichen `7b66b5664c31` → `0576685c0ecf`, Runs 2 und 4 die umgekehrte Richtung.
- Lokal vorhandenes CI-Artefakt: `visual-regression-report-2` aus **Run 1** (current = `0576685c0ecf`, `run_id` 40353105556). Das Artefakt von Run 4 liegt lokal nicht vor.
- Lokale FF3-Läufe: `runs/A1`, `runs/B5`, Auswertung `results/B5_vs_A1`.
- Schwelle für den Pixelabgleich in dieser Prüfung: maximale Kanaldifferenz > 38 (wie `threshold` im CI-Bericht).

## Ergebnis

1. Die Box-Zahlen sind in allen vier CI-Läufen gleich (triage_option_page 8/9, scan_page 11/12, activation_token_page 14/14, jeweils cv/Pixel).
2. Die CI-Screenshots von `0576685c0ecf` sind für alle vier Seiten innerhalb der Schwelle identisch mit A1 (0 abweichende Pixel).
3. Die Abweichung CI gegen B5 ist pixelgleich mit der Abweichung A1 gegen B5. Die CI-Differenz entspricht also dem Unterschied zwischen A1 und B5.
4. Die vier Seiten zeigen inhaltlich Seiten, die FF3 bereits auswertet:

| CI-Seite | Inhalt | Identisch mit (A1 und B5) | CI cv/Pixel | FF3 B5_vs_A1 cv/Pixel | Ursache der Boxen |
|---|---|---|---|---|---|
| `scan_page` | Startseite | `01_home_tab` | 11/12 | 11/12 | D11, D14, D16 und Folgeverschiebung (Auswahlfeld, „Starte die Triage“) |
| `activation_token_page` | Einstellungsseite, Token-Dialog nicht geöffnet | `language_button` | 14/14 | 14/14 | D03, D10, D15 und Layout-Verschiebung aus Abschnitt 13 |
| `language_choice_page` | Sprachdialog | `13_language_choice` | 20/14 | 20/14 | D08 und weitere Abweichungen (siehe FF3-Auswertung) |
| `triage_option_page` | Startseite mit geöffnetem Auswahlmenü | `03_triage_option` (nicht unter den 24 Vergleichen) | 8/9 | – | D11, D14, D16 und Folgeverschiebung. Keine Boxen im Menü |

## Folgerungen

- Die Abweichungen dieser Seiten gehen weder auf einen anderen Bildinhalt noch auf eine neue Verschiebung zurück. Sie entsprechen den Defekten D03, D08, D10, D11, D14, D15 und D16 sowie der Layout-Verschiebung aus Abschnitt 13.
- Abschnitt 2 Punkt 7 ist zu korrigieren: `scan_page`, `activation_token_page` und `language_choice_page` sind keine Kontrollseiten ohne Defekt.
- Die CI-Seiten dürfen nicht zu den 24 Seitenvergleichen addiert werden. Sonst werden Defekte doppelt gezählt.
- Offen: Die Richtung der Labels in `summary_report.json` (`baseline_label`, `actual_label`) ist am Artefakt von Run 4 zu bestätigen.

## Dateien

- `abgleich.csv`: Identitätsprüfung und Pixelzahlen je Seite
- `bilder/vergleich_<seite>.png`: A1 | B5 | CI (Run 1) nebeneinander
- `bilder/ci_diff_<seite>.png`, `bilder/ci_diff_<seite>_pixel_matching.png`: Differenzbilder aus dem CI-Bericht (Run 1, verkleinert)

![scan_page](bilder/vergleich_scan_page.png)
![activation_token_page](bilder/vergleich_activation_token_page.png)
![triage_option_page](bilder/vergleich_triage_option_page.png)
![language_choice_page](bilder/vergleich_language_choice_page.png)
