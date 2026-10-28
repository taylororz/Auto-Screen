# FF3 Protokoll: Versionsvergleich mit eingebrachten Darstellungsfehlern

Stand: 25.10.2026 (Aktualisierung nach Auswertung B1 vs. A1). Dieses Protokoll dokumentiert die Entscheidungen, Commits und Abweichungen vor den Laufzeiten A1, A2 und B1.

## 1. Versionen

| Komponente | Referenz | Bemerkung |
|---|---|---|
| Auto-Screen (Evaluationscode) | Tag `eval-final` → `b119068` | Die Auswertung verwendet `compare_images.py` im Stand dieses Commits (Blob `8d03680373e6`). Nicht gepusht. Nachtrag 26.10.2026: In der Arbeitskopie wurde nur der Text des HTML-Berichts geändert. Bei gleichem Ordnernamen auf beiden Seiten heißt der Bericht jetzt „Version comparison“. Vergleichslogik, Schwellenwerte und Zahlen sind unverändert (geprüft: 58 bzw. 53 Boxen vor und nach der Änderung). Der Tag `eval-final` zeigt weiterhin auf `b119068`. Zusätzlich kam die Option `--version-baseline-root` hinzu. Sie vergleicht jede Sprache mit dem gleichnamigen Ordner des vorherigen Release und erzeugt einen eigenen Bericht. Die Pipeline erstellt damit zwei Berichte: den Sprachvergleich (`diff_results/`) und, wenn `PREVIOUS_SCREENSHOTS` gesetzt ist, den Versionsvergleich (`diff_results/version_comparison/`). Geprüft mit A1 gegen B5: DE 58 bzw. 53 Boxen, EN und ZH 0. |
| KatApp Build A | `fdb6f79` (Branch `feat/zh-localization`) | Originalversion + Chinesische Übersetzung (zh_CN, zh_TW). |
| KatApp Build B | `af44aec` (Branch `seeded-defects`), gebaut als **Build B5** | Build A + 14 Commits D01–D16 (ohne D07, D12). Neubau aus sauberem Worktree; die deutschen Texte von D11 und D14 sind verlängert (committete `de.json`). Build B1 ist überholt, siehe Abschnitt 12. |

Nicht commitet, aber in beiden Builds identisch (siehe Abschnitt 4):
`flavors.dart` (API-Adresse `127.0.0.1:9997`), iOS-Deployment-Target, Podfile.

## 2. Entscheidungen

1. **Eval-Tag**: `eval-final` wurde auf `b119068` gesetzt. Das Verzeichnis `~/Masterarbeit/Auto-Screen` ist kein Git-Repository; die Experimente verwenden `~/Auto-Screen`.
2. **Build A**: nur die Chinesische Übersetzung wurde committet (Ausschluss: Dev-API-URL, Standort-Schlüssel-Migration, Casualty-Texte, iOS-Build-Dateien).
3. **Anzahl Defekte: 14 statt 20.** Begründung: Nur Texte, die nicht zur Navigation gehören und deren Layout ohne Änderung in EN/ZH bleibt, können gezielt verändert werden. Mit der Vorgabe "höchstens 3 Defekte je Seite" ergibt sich eine Obergrenze von 14. Die Verteilung ist: Gekürzter Text 4 (D01–D04), Beschnittener Text 3 (D05, D06, D08), Überlappung 3 (D09–D11), Verdrängtes Element 4 (D13–D16).
4. **Entfallen**: D07 (Login-Token-Label, `labelText` lässt sich nicht kürzen oder beschneiden) und D12 (Login-Titel-Überlappung; nur ein Textelement auf der Seite verfügbar). Die IDs bleiben zur Rückverfolgung erhalten.
5. **Teilweises Committen**: `triageOrtSelection.dart` enthält die Migration der Standortschlüssel. Die Commits D11 und D14 enthalten nur die jeweilige Textänderung.
6. **Seitenzuordnung**: Login-Seite: `login_screen.dart`; Startseite: `home_screen.dart` und `triageOrtSelection.dart` (page `triage_location_menu`); Sprachdialog: `setting_screen.dart`. Die Seitenzuordnung für `home_slogan` (D16) ist nicht durch einen Screenshot bestätigt.
7. **Unbenutzte Seiten** (Kontrolle ohne Defekte): `scan_page`, `activation_token_page`, `language_choice_page`.

## 3. Defekt-Tabelle (Build B, `351e8cb`)

| ID | Klasse | Seite | ARB-Key (de.json) | Commit |
|---|---|---|---|---|
| D01 | Gekürzter Text | overview_page | triages_overview_screen_title | 34cfe35 |
| D02 | Gekürzter Text | vehicle_page | vehicle_screen_last_sent | 49d72ab |
| D03 | Gekürzter Text | settings_page | setting_screen_data_privacy | 16b02dc |
| D04 | Gekürzter Text | login_page | login_screen_title | d4315b2 |
| D05 | Beschnittener Text | overview_page | no_upload_queued | 35042a2 |
| D06 | Beschnittener Text | vehicle_page | vehicle_screen_description | cc84cd3 (Beschreibung Fahrzeug-Tracking, dritte Zeile abgeschnitten) |
| D08 | Beschnittener Text | language_dialog | choose_language_home_screen | 9e4f6ae |
| D09 | Überlappung | vehicle_page | vehicle_screen_tracking_active | 7a4b2d8 (nur bei aktivem Tracking sichtbar, siehe Abschnitt 10) |
| D10 | Überlappung | settings_page | setting_screen_access_data | 953bd8a |
| D11 | Überlappung | triage_location_menu | home_select_current_triage_location | 066f5d1 |
| D13 | Verdrängtes Element | overview_page | triages_overview_screen_catastrophe | 7058b23 |
| D14 | Verdrängtes Element | triage_location_menu | home_current_triage_location | 351e8cb |
| D15 | Verdrängtes Element | settings_page | settings_screen_appearance_mode | f705ab8 |
| D16 | Verdrängtes Element | triage_location_menu | home_slogan | 8cfd9a2 |

Alle deutschen Texte stehen nur in `de.json`. EN- und ZH-Dateien wurden für diese Defekte nicht verändert.

## 4. Build-Umgebung (gilt für A und B gleich)

- Xcode 27. Das Deployment-Target wurde von 13.0 auf 15.0 angehoben (Podfile, `post_install`, `Runner.xcodeproj`), weil Xcode 27 Simulator-Builds ab 15.0 verlangt. Die Änderung ist nicht commitet.
- `flutter config --no-enable-swift-package-manager` (globale Flutter-Einstellung, CocoaPods statt SPM).
- `flutter build ios` scheitert am generischen Ziel "Any iOS Simulator Device" (Exit 255). Der Build erfolgt deshalb mit `xcodebuild -scheme dev -configuration Debug-dev -destination 'platform=iOS Simulator,name=iPhone 18 Pro'`, das Ergebnis wird nach `build/ios/iphonesimulator/Runner.app` kopiert. Bundle-ID: `de.kfru.katapp.dev` (entspricht `katapp.json`).
- Build B wurde erfolgreich gebaut. Build A ist noch nicht gebaut.

## 5. Prüfungen und Ergebnisse

| Prüfung | Stand |
|---|---|
| `flutter analyze` (geänderte Dateien) | Keine Fehler; drei bestehende Hinweise |
| Build A (`fdb6f79` + gleiche Konfiguration, ohne Defekte) | Erfolgreich (`xcodebuild dev`) |
| Build B2 (`af44aec`, Defekte nur in Deutsch aktiv) | Erfolgreich (`xcodebuild dev`) |
| Läufe A1, A2, B1 | Vollständig (4 Sprachen, 0 Fehler, Training-Daten nach jedem Lauf wiederhergestellt) |
| EN_US, ZH_CN, ZH_TW: B1 vs. A1 (ohne Login-Screenshots) | Identisch: 33/33, 107/107, 33/33 Seiten, Differenz 0 |
| Rauschkontrolle A1 vs. A2 (EN/ZH) | Differenz 0 |
| DE: B1 vs. A1 (ohne Login-Screenshots) | 4 von 36 Seiten identisch; Maximum 4,2 % Pixel |
| Gelb-Schwarz-Streifen (Warnung) | Keine; die gelben Flächen (Video-Seite, Bluesky-Menü) sind in A1 und B1 gleich |
| Navigation | In allen Läufen durchgehend ausgeführt |

### Ausgeschlossene Screenshots (Beschluss)

Die drei folgenden Screenshots werden aus allen Vergleichen ausgeschlossen:
`login_page.png`, `login_save_button.png`, `10_login_save_button.png` (in allen Sprachordnern).

Grund: Beim Screenshot nach der Token-Eingabe war in B1 die iOS-Tastatur sichtbar, in A1 nicht. Das ist ein Zeitfaktor des Laufs und keine Eigenschaft des Defekts. Ohne den Ausschluss würden diese Seiten in EN, ZH-CN und ZH-TW fälschlich als Abweichung erscheinen.

Folge für die Auswertung: D04 (Titel der Login-Seite) liegt im oberen Bereich der Seite und wird durch den Ausschluss nicht betroffen, weil Titel und Tastatur getrennt sind. Die Auswertung der Login-Seite für D04 muss trotzdem gesondert geprüft werden. Die Abgrenzung wird im Abschnitt 6 dokumentiert.

## 6. Abweichungen vom Versuchsplan (FF3_Experiment)

- 14 statt 20 Defekte; Begründung in Abschnitt 2.
- Klasse Überlappung: Die Defekte D09–D11 erzeugen Überlappungen nur über den Textumbruch (feste Höhe, `overflow: visible`), nicht über Überlagerung mit einem Icon.
- Klasse Verdrängtes Element: D13–D16 sind feste Breiten mit horizontalem, nicht bedienbarem ScrollView. Der Text wird abgeschnitten, nicht verschoben. Das ist eine Annäherung an das Konzept.
- Die Gruppe ist klein (14); Aussagen zu Recall je Klasse beruhen auf 3–4 Defekten und sind nur eingeschränkt belastbar.

## 7. Offene Schritte

1. Build A aus `fdb6f79` mit denselben Konfigurationsänderungen (Podfile, Deployment-Target, `flavors.dart`) bauen; dabei keine Defekt-Commits übernehmen.
2. Screenshots von A und B in EN, ZH-CN, ZH-TW und DE für die 10 Seiten erstellen.
3. Prüfen: EN/ZH in B identisch zu A; DE-Defekte sichtbar; keine Streifen; Navigation funktioniert.
4. Läufe A1, A2 (A) und B1 (B) mit Auto-Screen; `seeded_defects.json` mit den tatsächlichen Bounding-Boxen füllen (`bbox_picker.py`); `ff3_eval.py` ausführen.

## 8. Läufe

| Lauf | Build | Ergebnis | Ordner |
|---|---|---|---|
| A1 | A (`fdb6f79`) | Vollständig, 223 Dateien | `runs/A1/` |
| A2 | A (`fdb6f79`) | Vollständig | `runs/A2/` |
| B1 | B2 (`af44aec`) | Vollständig | `runs/B1/` |
| B1_v1_ungated | B (`351e8cb`) | Ersetzt: EN/ZH nicht identisch, D11/D14 fehlten im Build | `runs/B1_v1_ungated/` |

Der Lauf B1_v1 ist nicht Teil der Auswertung. Er wird nur dokumentiert, damit die Änderung nachvollziehbar bleibt.

Verifikation: `ff3/results/verification_pixel_diffs.csv` (alle Paare, mit Login-Screenshots; die Ausschlüsse oben gelten bei der Auswertung).

## 10. Manuelle Annotation (Ground Truth)

Die Bounding-Boxen wurden manuell mit `bbox_picker.py` auf dem B1-Screenshot gezogen, mit A1 als Referenz nebeneinander. Die Boxen liegen in `ff3/bbox_log/bboxes_settings.txt`, die Ground Truth in `ff3/seeded_defects.json`.

Die Zuordnung „Dateiname → Seite“ folgt dem Inhalt der Datei, nicht dem Namen. Mehrere Dateinamen zeigen dieselbe Seite (MD5 gleich) oder eine andere als ihr Name nahelegt. Beispiele: `overview_page.png` zeigt die Scan-Seite, `settings_page.png` zeigt die Startseite, `04_scan_tab.png` zeigt die Startseite. Für die Auswertung wird der Dateiname verwendet, unter dem der Defekt tatsächlich sichtbar ist.

| Defekt | Datei (Seite) | Box `[x, y, w, h]` | Status |
|---|---|---|---|
| D01 | `06_vehicle_tab.png` (Übersicht) | [276, 198, 643, 145] | markiert |
| D03 | `language_button.png` (Einstellungen) | [326, 1861, 579, 125] | markiert |
| D04 | `09_login_field.png` (Login) | [320, 198, 527, 125] | markiert |
| D06 | `07_settings_tab.png` (Fahrzeug-Tracking) | [61, 439, 1092, 168] | markiert |
| D08 | `13_language_choice.png` (Sprachdialog) | [171, 958, 739, 166] | markiert |
| D10 | `language_button.png` (Einstellungen) | [78, 1293, 792, 125] | markiert |
| D15 | `language_button.png` (Einstellungen) | [227, 751, 748, 148] | markiert |
| D16 | `01_home_tab.png` (Startseite) | [34, 218, 1153, 201] | markiert |
| D02 | `07_settings_tab.png` | leer | nicht identifiziert |
| D05 | `06_vehicle_tab.png` | leer | nicht identifiziert |
| D09 | `07_settings_tab.png` | leer | nicht identifiziert |
| D11 | `01_home_tab.png` (Titel „Wähle den Triage-Ort“) | [96, 858, 1014, 99] | aus Elementposition (Appium page source, ×3), Defekt visuell kaum wahrnehmbar |
| D13 | `06_vehicle_tab.png` | leer | nicht identifiziert |
| D14 | `01_home_tab.png` (Label „Aktuell ausgewählt:“) | [96, 1014, 405, 60] | aus Elementposition (Appium page source, ×3), Defekt visuell kaum wahrnehmbar |

**Korrektur:** Die Box, die zunächst als D09 geführt wurde, gehört zu D06. Die Textkürzung (dritte Zeile abgeschnitten) entspricht der ClipRect-Höhe 40 von D06. D09 ist ein Text für das aktive Tracking und war im Lauf nicht sichtbar.

**Nicht identifiziert (leere GT):** Die Defekte D02, D05, D09, D11, D13 und D14 waren im B1-Lauf im Screenshot nicht sichtbar oder nicht manuell erkennbar. D02, D05, D09 und D13 brauchen einen Zustand, der im Lauf nicht erreicht wurde (gesendete Position, aktives Tracking, wartende Uploads, Katastrophenmodus). D11 und D14 sind auf dem Startseiten-Screenshot nicht erkennbar. Die leere GT zählt in der Auswertung als nicht gefunden. Die Entscheidung dazu ist in Abschnitt 11 begründet.

**Nachtrag D11, D14 (Ergänzung nach Auswertung):** Die Frames stammen aus der Appium-Page-Source des installierten Build B (`builds/B2`). Die Zuordnung der ARB-Keys ist: D11 = `home_select_current_triage_location` (Titel), D14 = `home_current_triage_location` (Label). Die früher für D14 verwendete Box „Schadensort“ war falsch und ist verworfen.

**Befund zum Build:** Im Build B1 sind die deutschen Texte für beide Keys **nicht verlängert**. Die Arbeitskopie von `de.json` enthält die Originaltexte aus Build A (`Aktuell ausgewählt:`, `Wähle den Triage-Ort`). Die committete Fassung in `af44aec` enthält die verlängerten Texte. Der geprüfte Build B1 entspricht also nicht vollständig `af44aec`. Die Defekt-Layouts D11 und D14 sind im Code vorhanden, zeigen aber keinen sichtbaren Effekt. Die Commits 066f5d1 und 351e8cb sind im Build B1 wirkungslos.

**Prüfung über Pixel in den Boxen:** Die Ausschnitte in `ff3/gt_check/D11.png` und `D14.png` zeigen für D14 identischen Text, nur um einige Pixel verschoben. Die Differenz in der Box beträgt 6,45 % (D11) und 22,77 % (D14). Diese Werte entstehen durch die Layout-Verschiebung aus Abschnitt 13, nicht durch den Defekt.

## 11. Entscheidung zur Stichprobe

Vorschläge, die sechs schwer sichtbaren Defekte aus der Auswertung zu entfernen oder durch andere zu ersetzen, wurden verworfen. Entfernen würde die Recall-Werte auf die leicht sichtbaren Fälle beschränken. Ersetzen nach Sichtung der Ergebnisse wäre eine nachträgliche Auswahl. Die 14 Defekte bleiben die Grundgesamtheit. Berichtet werden:
- Recall über alle 14 Defekte,
- Recall über die 8 Defekte mit manueller Box, mit dem Hinweis, dass dies eine Teilmenge der leicht sichtbaren Defekte ist.

Ein Zusatzexperiment mit neuen Defekten wäre nur mit vorab festgelegtem Protokoll zulässig und müsste als „nachträglich ergänzt“ gekennzeichnet werden.

## 12. Auswertung B1 gegen A1 (DE)

**Version 2 (aktuell, nach Ergänzung von D11 und D14):** Die Ergebnisdatei `summary.json` ist die neue Version. Die alte Fassung heißt `summary_v1.json`, die alte Seitenliste `page_results_v1.csv`. Die Differenzbilder in `diffs/` wurden überschrieben.

| Methode | Recall | Box-TP | Box-FP | Precision | Nicht gefunden |
|---|---|---|---|---|---|
| Pixel | 10/14 (71,4 %) | 15 | 38 | 28,3 % | D02, D05, D09, D13 |
| Grau (cv) | 10/14 (71,4 %) | 14 | 44 | 24,1 % | D02, D05, D09, D13 |

**Wichtig:** Die zwei zusätzlichen Treffer D11 und D14 sind keine Erkennung des Defekts. Die Boxen `[94, 1020, 1018, 99]` und `[95, 876, 340, 61]` liegen auf der Layout-Verschiebung. Der Text ist in B1 unverändert. Der Recall von 10/14 ist deshalb überhöht. Der belastbare Wert ist 8/14 (Version 1, Abschnitt unten). Die beiden Werte müssen zusammen berichtet werden.

**Version 1 (vor der Ergänzung):** Pixel 8/14 (57,1 %), Grau 8/14 (57,1 %), Precision 22,6 % bzw. 19,0 %.

Befehl (Seiten explizit, weil die Standardliste in `ff3_eval.py` nicht den Dateinamen entspricht):
```
python3 ff3_eval.py --tool-dir ~/Auto-Screen --baseline runs/A1/screenshots --actual runs/B1/screenshots --gt seeded_defects.json --out results/B1_vs_A1 --pages 06_vehicle_tab 07_settings_tab 01_home_tab language_button login_field 13_language_choice
```
Die Kontrolle A2 gegen A1 wurde mit denselben Seiten ohne `--gt` ausgeführt.

Tabelle Version 1 (überholt, Dateien `summary_v1.json` und `page_results_v1.csv`):

| Vergleich | Methode | Recall | Boxen gesamt | Box-TP | Box-FP | Precision | Boxen auf Seiten ohne Defekt |
|---|---|---|---|---|---|---|---|
| B1 vs. A1 | Pixel | 8/14 (57,1 %) | 53 | 12 | 41 | 22,6 % | 0 |
| B1 vs. A1 | Grau (cv) | 8/14 (57,1 %) | 58 | 11 | 47 | 19,0 % | 0 |
| A2 vs. A1 | Pixel / cv | – | 0 | – | – | – | 0 |

Recall nach Klasse (Pixel): Beschnittener Text 2/3, Gekürzter Text 3/4, Verdrängtes Element 2/4, Überlappung 1/3. Nicht gefunden: D02, D05, D09, D11, D13, D14.

Die Zahlen gelten für 24 Seitenvergleiche (6 DE-Seiten mit Defekt, 4 Ordner, 24 Paare insgesamt). Die Seiten ohne Defekt sind EN und ZH sowie die übrigen DE-Seiten.

## 13. Befund: Layout-Verschiebung in DE (Nebenwirkung der Implementierung)

Die Differenzbilder zeigen Boxen auf Elementen ohne Codeänderung, zum Beispiel auf den Überschriften „Erscheinung“, „System“ und „Info“ und auf den Buttons „Sprache“, „Datenschutz“ und „Benutzerhandbuch“ (Seite `language_button`).

Gemessen mit Template-Matching zwischen A1 und B1 (gleiche Seite, DE):
- „Erscheinung“: dy = −6 px
- „System“: dy = −6 px
- „Sprache“-Button: dy = −6 px
- „Info“: dy = +6 px

Die Verschiebung ist vertikal und hat in beiden Richtungen unterschiedliche Vorzeichen. Die Ursache ist vermutlich die Änderung der Höhe eines Kindelements in einer `Column` mit `Spacer`. Die Vermutung ist noch nicht im Code geprüft.

Die Verschiebung tritt nur in DE auf, weil die Defekte nur dort aktiv sind. EN und ZH sind identisch (Abschnitt 5).

Folgen:
- Die 41 Pixel-Fehlalarme enthalten einen erheblichen Anteil dieser Verschiebung. Die Precision ist daher nicht allein dem Detektor zuzuschreiben.
- Die „sauberen“ DE-Seiten sind in B nicht vollständig sauber. Das ist eine Validitätsbedrohung der Auswertung und muss in der Arbeit genannt werden.

Mögliche Abhilfe: Die Defekt-Wrapper behalten die Höhe des Originalelements. Dies würde Build B ändern und einen neuen Lauf B1 erfordern (Abschnitt 14).

## 14. Versuch der Layout-Korrektur (B3, B4): nicht erfolgreich

Ziel war, dass die DE-Defekte die Höhe der Nachbarelemente nicht verändern (Abschnitt 13).

| Version | Commit (katapp) | Helper | Lauf | Ergebnis |
|---|---|---|---|---|
| B1 / B2 | `af44aec` | Ternärer Ausdruck: DE zeigt Defekt, sonst Original | `runs/B1` | Vertikale Verschiebung ±6 px (Einstellungen), EN/ZH identisch zu A1 |
| B3 | `f611076` | Stack: Original mit `Visibility(maintainSize)` als Layout-Box, Defekt in `OverflowBox` | `runs/B3_fix` | EN/ZH identisch; DE-Verschiebung auf ±36 px gestiegen |
| B4 | `dd4b6fe` | Stack: Original mit `Opacity(0)` + `IgnorePointer` + `ExcludeSemantics` | `runs/B4_fix` | Screenshots identisch zu B3 (MD5 gleich); gleiche Verschiebung |

Befund: Die Art der Platzhalter (`Visibility` oder `Opacity`) ist nicht die Ursache, denn B3 und B4 erzeugen identische Screenshots. Die Verschiebung entsteht mit der Stack-Struktur, die in B2 nicht vorhanden war. Die genaue Ursache wurde nicht ermittelt.

Entscheidung: Die Korrektur wird nicht weiter verfolgt. Jeder Versuch kostet einen Build und einen Lauf von insgesamt etwa 12 Minuten, und jeder Versuch änderte nur eine Variable. Die offizielle Build-B-Version für die Auswertung bleibt B1 (Quellstand `af44aec`, Helper als ternärer Ausdruck). B3 und B4 werden nicht ausgewertet. Die Commits `f611076` und `dd4b6fe` wurden durch `git reset --keep af44aec` aus dem Branch `seeded-defects` entfernt (HEAD = `af44aec`). Sie sind nur noch im Reflog erreichbar (`dd4b6fe`). Die Arbeitskopie-Änderungen blieben erhalten.

Konsequenz für die Arbeit: Die DE-Ergebnisse werden mit der Verschiebung aus Abschnitt 13 berichtet. Die Verschiebung betrifft auch DE-Seiten ohne Defekt und senkt die Precision. Sie wird als Validitätsbedrohung genannt.

## 16. Neubau Build B5 und Auswertung B5 gegen A1

**Anlass:** Im Build B1 waren die deutschen Texte von D11 und D14 nicht verlängert (Arbeitskopie von `de.json`). Die Defekte waren daher nicht sichtbar. Abschnitt 12 beschreibt das.

**Vorgehen:** Worktree aus `af44aec`, dazu die übrigen unkommitteten Build-Änderungen (ohne `de.json`), `pod install` und `flutter pub get`. Zwei Pfadfehler traten auf: `Generated.xcconfig` und `flutter_export_environment.sh` zeigten noch auf das Hauptverzeichnis. Die Assets kamen dadurch aus der falschen Quelle. Nach der Korrektur enthält die App die verlängerten Texte.

**Prüfung:** Im Bundle von B5 steht `Aktuell ausgewählter Triage-Ort der Einsatzstelle:` und `Wähle den Triage-Ort der aktuellen Einsatzstelle`. Im Screenshot von B5 ist D14 als abgeschnittener Text sichtbar. D11 zeigt „Wähle den Triage-Ort der“ und bricht danach ab.

**Lauf:** B5 (Lauf `runs/B5`), Restore der Trainingsdaten: RESTORE_OK.

**Auswertung** (gleicher Befehl wie Abschnitt 12, `--actual runs/B5/screenshots`, Ausgabe `results/B5_vs_A1`):

| Methode | Recall | Nicht gefunden | Box-TP | Box-FP | Precision |
|---|---|---|---|---|---|
| Pixel | 10/14 (71,4 %) | D02, D05, D09, D13 | 15 | 38 | 28,3 % |
| Grau (cv) | 10/14 (71,4 %) | D02, D05, D09, D13 | 14 | 44 | 24,1 % |

Recall nach Klasse (Pixel): Beschnittener Text 2/3, Gekürzter Text 3/4, Verdrängtes Element 3/4, Überlappung 2/3.

**Einschränkung:** Auf der Startseite überlappen die Boxen von D11 und D14 teilweise mit den Boxen der Layout-Verschiebung aus Abschnitt 13. Ob die Erkennung allein auf den Defekt zurückgeht, lässt sich mit den Boxen nicht vollständig trennen. Die Layout-Verschiebung ist in B5 weiterhin vorhanden, weil B5 denselben Helper wie B1 verwendet. Die Ergebnisse von B5 müssen mit diesem Vorbehalt berichtet werden.

**EN, ZH:** B5 gegen A1 identisch (EN 31 identisch, ZH_CN 103, ZH_TW 32; keine Abweichung).

Die Ergebnisse von B1 (Abschnitt 12) bleiben als Dokumentation erhalten, werden aber nicht mehr als Build B berichtet.

## 15. Nächste Schritte

1. B1 gegen A1 mit den Ergebnissen aus Abschnitt 12 als Hauptergebnis berichten. Die Verschiebung aus Abschnitt 13 in der Diskussion nennen.
2. Entschieden: D04 (Login-Titel) wird in der DE-Auswertung berücksichtigt. Die Box liegt auf `09_login_field.png`, das ohne Tastatur ist. Die Seite `login_field` ist in der Auswertung enthalten; in EN und ZH erzeugt sie keine Boxen.
3. Build B ist `af44aec`. Die Commits `f611076` und `dd4b6fe` werden nicht erwähnt, außer in der Protokoll-Historie.
