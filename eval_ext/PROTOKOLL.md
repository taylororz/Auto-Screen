# Ergänzende Auswertung der optionalen Funktionen (nach der Evaluation)

Stand: 28.10.2026. Code der Funktionen: Zweig `feat/improvements`, Commit `2f50706`.
Die evaluierte Fassung der Arbeit ist `eval-final-2` (`2d2693e`). Diese Auswertung ändert keine Zahl der Hauptevaluation.

Dieses Protokoll wurde **vor** den Läufen geschrieben und committet. Daten, Parameter und Kennzahlen stehen damit vor den Ergebnissen fest.
Nach den Läufen werden Parameter nicht geändert. Abweichungen werden unten unter „Abweichungen“ dokumentiert.

Durchführung: Skripte und Läufe mit Unterstützung von Claude Code (Opus 5.5). Die Ergebnisse prüft der Verfasser.

## Vorab bekannte Einsichten (Offenlegung)

- Bei der Entwicklung von E2 wurde die lokale Einstufung an zwei Seiten aus Lauf B5 ausprobiert (DE `07_settings_tab`, vier Boxen, und DE `06_vehicle_tab`, eine Box).
  Dabei wurden zwei Varianten (größerer Rand, genauerer Prompt) getestet und verworfen. Prompt und Rand blieben wie in `2f50706`.
- E1 wurde mit drei Werten (τ = 30, 38, 46, Mindestgröße 14) bereits als Funktionstest ausgeführt. Ergebnis damals: 10/14 bei allen drei Werten.
- E3 wurde an zwei Screenshots aus FF3 (nicht aus iOS-1K) als Funktionstest ausgeführt.

## E1: Empfindlichkeit der Schwellenwerte

- Frage: Wie stark hängen die FF3-Ergebnisse von τ = 38 und der Mindestgröße 14 px ab?
- Daten: FF3, B5 gegen A1 (mit `ff3/seeded_defects.json`) und A2 gegen A1 (Kontrolle ohne Fehler). Ordner DE, EN, ZH_CN, ZH_TW. Seiten wie in FF3:
  `06_vehicle_tab 07_settings_tab 01_home_tab language_button login_field 13_language_choice`.
- Raster: τ ∈ {20, 26, 30, 34, 38, 42, 46, 50}, Mindestgröße ∈ {10, 12, 14, 16, 20} px. Masken 140/70 px.
- Kennzahlen je Methode: gefundene Fehler (von 14), Boxen gesamt, Box-Precision, Boxen in der Kontrolle.
- Auswertung als Empfindlichkeitsanalyse. Es werden **keine** neuen Standardwerte gewählt (FF3 berichtet auf denselben Daten).
- Befehl: `python3 tools/calibrate_thresholds.py --set ff3/runs/A1/screenshots ff3/runs/B5/screenshots ff3/seeded_defects.json --set ff3/runs/A1/screenshots ff3/runs/A2/screenshots none --pages <Seiten> --out eval_ext/results/e1`

## E2: Lokale Einstufung der Differenzboxen

- Frage: Trennt ein lokales VLM ohne Feintuning die Boxen eingebrachter Fehler von übrigen Boxen? Wie viele Boxen müsste eine Person danach noch ansehen?
- Modell: `Qwen/Qwen3-VL-4B-Instruct` (Basismodell, lokal, ohne Feintuning). Prompt und Rand (24 px) wie in `common_agent/diff_triage.py` (`2f50706`). Bewertet werden **alle** Boxen (keine Obergrenze).
- Boxen: Graustufenabgleich (wie die Einstufung im Bericht), τ = 38, Mindestgröße 14.
  - V: Versionsvergleich B5 gegen A1, vier Ordner, FF3-Seiten. Fehlerbox = Box schneidet ein annotiertes Rechteck (Regel aus `ff3_eval.py`).
  - L: Sprachvergleich in A1, DE gegen EN, ZH_CN und ZH_TW, FF3-Seiten. Alle Boxen sind erwartete Übersetzungsunterschiede (Build A enthält keine eingebrachten Fehler).
- Kennzahlen:
  1. Fehlerboxen in V: Anteil `defect` sowie Anteil `defect` oder `unclear`.
  2. Fehler (von den in FF3 gefundenen): Anteil mit mindestens einer Box `defect` bzw. `defect` oder `unclear`.
  3. Übrige Boxen in V und alle Boxen in L: Anteil `acceptable`.
  4. Aufwand: Anteil aller Boxen (V und L), die als `acceptable` übersprungen werden könnten, und Zahl der Fehler, die dabei verloren gingen.
- Befehl: `python3 eval_ext/e2_triage.py --out eval_ext/results/e2`

## E3: Teilbild-Verfeinerung bei Direct Grounding

- Frage: Trifft das feinabgestimmte Modell mit der Verfeinerung nach Ferret-UI häufiger das Zielelement?
- Daten: bereinigter iOS-1K-Evaluationssplit wie in FF2 (Bilder nicht im Training, 2008 Fälle). Zufallsstichprobe von 500 Fällen mit Seed 42
  (gleiches Verfahren wie `ios1k_som_eval.py --n`). Eingabe ist der unmarkierte Originalscreenshot (828 × 1792 px).
- Modell: `Qwen_trained/Qwen3-VL-4B-V2` (wie in der Evaluation), DG-Prompt unverändert.
- Bedingungen je Fall: (a) ohne Verfeinerung, (b) mit Verfeinerung (Überlappung 0,1, maximale Verschiebung 0,15 der Bildhöhe).
  (b) nutzt die Vorhersage von (a) als ersten Schritt. Fälle mit Rückfallebene (keine Koordinate) werden nicht verfeinert.
- Ground Truth: Mittelpunkt des Zielelements aus dem iOS-1K-Split (`gt_x`, `gt_y`). Die Element-Rechtecke liegen nicht vor.
- Kennzahlen:
  1. Primär: Treffer bei Abstand ≤ 40 px (Regel der Arbeit), Wilson-Intervall, exakter McNemar-Test (gepaart).
  2. Sekundär: Treffer bei ≤ 27 px (entspricht 40 px bei Faktor 3, umgerechnet auf Faktor 2), Median des Abstands, je Elementart.
  3. Anteil übernommener Verfeinerungen, Rechenzeit je Fall.
- Befehle: `python3 eval_ext/e3_dg_subimage.py run --n 500 --out eval_ext/results/e3.jsonl` und `... summary --run eval_ext/results/e3.jsonl --out eval_ext/results/e3_summary.json`

## E4: Freigegebene Basis (Demonstration)

- Frage: Was ändert sich, wenn die Pipeline gegen eine freigegebene Basis statt gegen den vorherigen Build vergleicht?
- Daten: Wiedergabe der abwechselnden CI-Folge mit lokalen FF3-Läufen: A1 → B5 → A2 → B5.
  - Fortlaufende Basis (bisher): B5 gegen A1, A2 gegen B5, B5 gegen A2.
  - Freigegebene Basis A1: B5 gegen A1, A2 gegen A1, B5 gegen A1.
- Kennzahlen: Boxen (Pixelabgleich), Boxen an annotierten Fehlerstellen, Boxen in Vergleichen, deren aktueller Build A ist (fehlerfrei).
- Das Ergebnis folgt aus der Logik und ist keine empirische Prüfung.
- Befehl: `python3 eval_ext/e4_baseline_replay.py --out eval_ext/results/e4`

## E3b: Teilbild-Verfeinerung in den Abläufen (Nachtrag nach E3, nicht vorab geplant)

Nachtrag vom 28.10.2026, geschrieben nach dem Ergebnis von E3 und vor dem Lauf von E3b.
Anlass: E3 fand keine Verbesserung auf iOS-1K (828 × 1792 px). Offen blieb, ob die Verfeinerung bei den
Screenshots der Abläufe (1206 × 2622 px), deren Auflösung sie stärker erhöht, schadet oder nützt.

- Daten: alle 598 manuell geprüften DG-Schritte des feinabgestimmten Modells aus FF2 (KatApp 262, Bluesky 336),
  `Qwen_trained/diff_results/finetune_dataset_DG.jsonl`. Die Screenshots liegen in der Arbeitskopie der Läufe vom 28.09.2026.
- Einschränkung vorab: Alle 598 Schritte waren richtig, und der Zielpunkt jedes Eintrags ist der bestätigte vorhergesagte Punkt.
  Eine Verbesserung der Trefferquote ist deshalb nicht messbar. E3b prüft nur, ob die Verfeinerung schadet.
- Ablauf: Erster Punkt ist der protokollierte und bestätigte Punkt. Darauf wird `_refine_in_subimage()` mit den Parametern
  aus `2f50706` angewendet (Überlappung 0,1, maximale Verschiebung 0,15). Ein erneuter erster Aufruf des Modells entfällt.
- Kennzahlen:
  1. Anteil übernommener Verfeinerungen, Verschiebung gegenüber dem bestätigten Punkt (Median, Anteil bis 40 px), je App.
  2. Schritte mit übernommener Verfeinerung und Verschiebung über 40 px prüft der Verfasser manuell am Screenshot
     (Ziel getroffen ja oder nein), wie in FF2. Schrittgenauigkeit mit Verfeinerung = (Schritte bis 40 px + manuell bestätigte) / 598.
  3. Rechenzeit der Verfeinerung je Schritt.
- Befehle: `python3 eval_ext/e3b_dg_flows.py run --out eval_ext/results/e3b.jsonl`, danach `... review` (Bilder für die
  manuelle Prüfung) und `... summary`.

## Nicht ausgewertet

- Stabilere Screenshots (Tastatur, Warten auf Ruhe): erfordert neue Läufe am Simulator.
- Geräte, dunkle Darstellung, Textgröße, RTL, physische Geräte: erfordert neue Läufe und eine manuelle Prüfung jedes Schritts.

## Abweichungen

- Bezeichnung: In der Arbeit heißen E1 bis E4 „EA1 bis EA4“ (Abschnitt „Ergänzende Auswertung der Erweiterungen“), weil E1 bis E6 dort die Entwurfsentscheidungen bezeichnen. Inhalt, Daten und Parameter sind unverändert.
- E1: Der Befehl wurde mit der FF3-Seitenliste aus diesem Protokoll ausgeführt. Keine Abweichung.
- E4: Keine Abweichung.
- E3: Keine Abweichung. 500 Fälle, keine Rückfallebene. Temporäre Teilbilder nach dem Lauf gelöscht.
- E3, nachträgliche Auswertung (nicht vorab geplant): Die Annahme „Element-Rechtecke liegen nicht vor“ war falsch. Sie stehen in den
  Annotationen von MUIAnno (`iOS-1K-Mobile-UI-Dataset/annotations/`). `e3_boxcheck.py` ordnet die Anweisungen über den Mittelpunkt
  (bis 3 px) eindeutig einem Rechteck zu (432 von 500, 36 ohne, 32 mehrdeutig) und zählt Punkte innerhalb des Rechtecks.
  Ergebnis: 72,2 % ohne, 73,4 % mit Verfeinerung, 5 gegen 10 abweichende Fälle, McNemar p = 0,30. Die vorab festgelegten
  Kennzahlen bleiben die primären.
- E3b: 598 Schritte, Verfeinerung in 590 übernommen (8 durch die maximale Verschiebung verworfen, alle `open_language_settings`).
  27 übernommene Verfeinerungen mit Verschiebung über 40 px (Bluesky: `go_to_Lists_page` 13, `go_to_Saved_page` 8,
  `open_language_settings` 6), alle nach rechts in derselben Menüzeile.
  Abweichung bei der manuellen Prüfung: Statt jeden Fall nur am Screenshot zu beurteilen, tippte der Verfasser je Schritt
  die Stelle eines verfeinerten Punkts in Bluesky an. In allen drei Schritten öffnete sich das Ziel. Da alle Punkte eines
  Schritts in derselben Zeile liegen, wurden alle 27 Fälle in `e3b_review/manual.json` als getroffen eingetragen.
  Ergebnis: 598/598, Rechenzeit der Verfeinerung im Median 3,7 s je Schritt.
- E2: Keine Abweichung. 296 Boxen (V 58, L 238), alle bewertet. Die Stufe `unclear` kam nicht vor, Kennzahlen „defect oder unclear“ sind daher gleich „defect“.
  Nachträgliche Beobachtung (keine vorab festgelegte Kennzahl): In drei Fällen nennt die Begründung einen abgeschnittenen Text bei der Einstufung `acceptable`.
