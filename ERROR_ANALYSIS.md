# Error analysis: dev set, whole pipeline (2026-11-15)

Every false positive and false negative of the whole system (back photo → rules → front photo → verdict) on the dev split, examined one by one on the photos: why it failed and what would fix it.

**Summary.**
- Same verdict as the hand labels: **80.7%**; back, all three warning decisions right: **90.4%**.
- Of the 20 verdict errors, **11 are app errors** (9 back, 2 front). **6 come from the hand labels** (wrong values, wrong status, a blank value, a label convention), and 3 are outside what the app covers (a Spanish table, a per-portion-only format, an exempt oil). Details in §8.
- The genuine back errors are mostly one problem: **assigning numbers to the right row** on curved, crumpled or offset tables (~10 cases across the analysis).
- The front had only 2 false negatives.

---

## 1. The run

- **Code:** branch `better-results` at `c2108a5`, default settings: `ENGINE = "combined"`, `FALLBACK_MODEL = "medium"`, front `yolo`.
- **What ran:** exactly the app's path (`web/app/pipeline.analyze`: `read_photos` + `summarize`) on **all 946 dev products**. The scoring uses the same truth as `src/evaluation/end_to_end.py` (`true_readings` + `summarize`).
- **No OCR cache was read:** every OCR call ran fresh, with 0 cache hits in the run. The results were *written* to a new cache, `data/cache/back_ocr_fresh_20261115/`, which replays the run exactly (0 misses, identical values).
- **Machine:** 8 workers on a 12-core local machine, about 24 min, no errors, median 8.1 s per product per worker.
- **Files** (`data/evaluation/e2e_dev_20261115_fresh/`):
  - `dev_run.jsonl`: one row per product, with the verdict, the values read and their sources, the row text matched, the photo type, rotation, and the front boxes and text.
  - `dev_run.pkl`: the same rows joined with the labels, as a DataFrame.
  - `debug/<code>/`: the back debug images (`01_original` … `08_parsed_values`) for every back case below.
  - `scripts/`: `run_dev.py` (the run), `analyze.py` (metrics), `cases.py` (case listing), `debug_cases.py` (debug images from the cache), `lines.py` (raw OCR lines of a product), `view.py` (crops).

### Definitions

- **Verdict level** (493 products with both sides labeled; positive = *non-compliant*):
  - **FP**: the app says non-compliant and the labels don't (compliant or inconclusive).
  - **FN**: the labels say non-compliant and the app doesn't (compliant or inconclusive).
- **Back level** (838 photos hand-labeled readable): per nutrient, warning required according to the reading vs according to the label.
- **Front level** (489 photos labeled "front"): per nutrient, warning found vs labeled.

## 2. Results

| Labels \ App | compliant | inconclusive | non-compliant |
| :--- | ---: | ---: | ---: |
| **compliant** (331) | **303** | 24 | 4 |
| **inconclusive** (106) | 51 | **46** | 9 |
| **non-compliant** (56) | 3 | 4 | **49** |

- **Same verdict as the labels: 80.7%** (493 products). When the app decides (not inconclusive, 419 products): **84.0%** right.
- Back, readable hand labels, all three decisions right (all three known, 726 photos): **90.4%**.

| Back, per nutrient | Known | Right | FP | FN | Unknown (of which: photo judged unreadable) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Added sugars | 796 | 95.9% | 3 | 3 | 27 (6) |
| Saturated fat | 786 | 95.0% | 5 | 4 | 30 (9) |
| Sodium | 784 | 93.1% | 0 | 7 | 47 (12) |

- Front: all three right on 98.4% of 489 photos. There were **0 FP**, **2 FN** (one saturated fat, one sodium), and **6 photos left unknown** (a panel detected but its nutrients not read).

### Photo type: hand label vs app

| Label \ App | linear | no_table | old_format | table |
| :--- | ---: | ---: | ---: | ---: |
| linear | 1 | 1 | 0 | 25 |
| no_table | 0 | 12 | 3 | **44** |
| ok | 2 | 5 | 7 | 778 |
| old_format | 0 | 1 | 40 | 8 |
| partial | 0 | 0 | 0 | 19 |

The 44 photos labeled `no_table` but read as a table stand out. The app read all three values on 28 of them, and the two checked by eye (5890500030011, 7898933095056) are perfectly readable tables, so the label is wrong.

---

## 3. Verdict false positives (13)

### 7891097013003: RapidOCR reads `0 g` as `60` (back, OCR)
- **Read:** saturated fat 60 g/100 ml, so the warning is required and missing. **True:** 0.
- **Why:** a clean digital table, but RapidOCR (small) read every `0 g` cell as `60`: total fat, saturated fat and trans fat alike (`'Gorduras saturadas (g) 60 60 0'`).
- **Why the existing guard didn't catch it:** total fat was misread as 60 too, so "part ≤ total" passed.
- **What contradicts it:** the energy row (62 kcal allows at most 6.9 g of fat) and the %VD column (`0`).
- **Fix:** energy bound and %VD consistency (§7.2).

### 7898571520156: curved package, rows shifted (back, row matching)
- **Read:** saturated fat 13 (true 3), sodium 0 (true 200).
- **Why:** on the curved surface the rows sag to the right. The "Gorduras saturadas" label was grouped with the total fat numbers `13 4 6`, and "Sódio" with the fibre row's `0 0 0`.
- **What contradicts it:** %VD 6 = 4 g ÷ 65 g, total fat's daily value; for saturated fat (20 g) it would be 20.
- **Label:** no basis marked, so the truth came out inconclusive. The table clearly says `100 g`; with the basis set, the truth would be compliant.
- **Fix:** §7.1, §7.2.

### 5010477352897: label column offset by half a row (back, row matching)
- **Read:** added sugars 24 (true 14).
- **Why:** on the whole photo, the label "Açúcares adicionados (g)" (cy 309, box height 62) had both `24` (cy 284, the total sugars value) and `14` (cy 321) within the grouping tolerance. `value_in_column` picks the candidate closest **in x**, and both are in the same column, so it took the first.
- **The fix in this case:** `14` is twice as close vertically (12 px vs 25 px). RapidOCR's label boxes are also ~1.7× taller than the number boxes, which widens the tolerance.
- **What contradicts it:** the row taken has no %VD, which is the signature of the total sugars row; added sugars has `11`.
- **Fix:** §7.1 (nearest vertically), §7.2.

### 7898311851410: curved cup, rows shifted (back, row matching)
- **Read:** added sugars 16 (true 13; total sugars' value), saturated fat 4.8 (true 3.1; total fat's value).
- **What contradicts it:** same signature as above; the row taken has no %VD, while added sugars has 36.
- **Label:** no basis marked; the table says `100 g`.
- **Fix:** §7.1, §7.2.

### 7891164005832: crumpled package, rows shifted (back, row matching)
- **Read:** sodium 0.6 (true 1260).
- **Why:** from trans fat down, every label sits one row above its numbers, so "Sódio" took fibre's `0,6`. The front correctly shows ALTO EM SÓDIO, so the app reported "has the warning but doesn't reach the limit".
- **Label:** saturated fat left blank (partly legible, 12).
- **Fix:** §7.1 (order alignment).

### 7898933095056: front misread `SÓO1O` (front, OCR)
- **Back:** read perfectly (saturated fat 10.5, sodium 801).
- **Front:** the panel's "SÓDIO" was read as `SÓO1O`, which scores 0.6 against `sodio`, below the crop threshold of 0.7.
- **Label:** the back is labeled `no_table`, but it is a perfectly readable table. With the right label, the truth would be compliant.
- **Fix:** §7.6 (map digits to look-alike letters: `1`→i, `0`→o gives `sooio`, which scores 0.8).

### 7896024762016: front panel with glare (front, image)
- **Why:** a horizontal panel whose "GORDURA SATURADA" cell is washed out by glare. Only `ADICUCNADO` was read, so saturated fat came out absent.
- **Back:** right. Sodium was read as 203 vs 293 on the label, harmless.
- **Fix:** §7.6 (count the panel's nutrient cells, so an unread cell gives "unknown", not "absent").

### 7896311768332: per-portion-only linear text (back, format)
- **Why:** the declaration only gives amounts **per portion** with %VD ("Valor energético (194 kcal, 10%) • … Gorduras saturadas (4,4 g, 22%) …"); it has no per-100 g values. The app treated them as per 100 g.
- **And:** Tesseract's `(4,4 9, 22%)` (the `g` read as `9`) gave 9 g, because `value_after` skips every token with a parenthesis as a unit.
- **Truth:** inconclusive, as the label has no basis. Converted with the portion size, the product may well need the warning.
- **Fix:** §7.5 (detect the per-portion-only format: convert with the portion, or abstain).

### 5890500030011: vegetable oil, Annex XVI (missing exemption + label)
- **Why:** the reading is right (saturated fat 14 g/100 ml). But oils are an Annex XVI **forbidden** category: they must *not* carry the warning. The app uses `Exemption.NONE`, so it demands one.
- **Label:** wrong: `no_table` on a clean table.
- **Fix:** §7.5 (oil detection from the table: ≥ ~90 g fat, 0 carbohydrates, 0 protein).

### 8445290480538: Spanish-only table (back, scope)
- **Why:** "Grasas saturadas" matched the key `saturadas`. The label note says: "readable but spanish: no recognition expected".
- **Fix:** §7.5 (a table without Portuguese words is not RDC 429 → inconclusive).

### 7891000368572: Spanish warning octagons on the front (label convention)
- **Why:** an imported chocolate whose front sticker has only the Spanish "EXCESO EN" octagons. The labeler marked the front `unreadable`, so the truth is inconclusive. The YOLO detector found no panel, so the app said non-compliant.
- **Arguably right for Brazil:** the Brazilian warning really isn't there.
- **Back:** sodium was read as 4.1 from the Spanish text, harmless.
- **Fix:** §7.5 (Spanish octagons → inconclusive), or agree on the label convention.

### 7896625211005: the label is wrong (app right)
- **Read:** saturated fat 14 g/100 g; **label:** 4.
- **The photo:** "Gorduras saturadas 14 g (4,3 g, 22%)" with a 30 g portion; 4.3 g per 30 g is 14.3 g per 100 g. The label dropped the 1.
- **With 14:** the labels' own verdict becomes non-compliant, the same as the app's.

### 7897685914516: label blank + warning off the front photo
- **Back:** added sugars 50 is perfectly legible, and the app read it; the label (`partial`) left it blank.
- **Front:** the "front" photo is the tub's lid, without a warning. The back photo shows the ALTO EM AÇÚCAR ADICIONADO panel on the tub's side.
- **So:** given the photos, the app's answer follows; the product itself is probably compliant.
- **Fix:** fill the label. Optionally, run the panel detector on the back photo too (§7.6).

---

## 4. Verdict false negatives (7)

### 7895000528448: faint value missed, portion column taken (back, column rule)
- **Read:** saturated fat 1.1 (true 7.4).
- **Why:** the 100 g value `7,4` is printed in faint gray and RapidOCR missed it. `first_right` then took `1,1` from the **15 g portion column**, although both headers had been read.
- **The fix in this case:** 1.1 × 100/15 = 7.3 g.
- **Note:** the only ingredient is "Amendoim torrado" (plain roasted peanuts), conditionally exempt under Annex XVI, so the labels' "non-compliant" is itself doubtful.
- **Fix:** §7.3, §7.2 (rebuild the value from the portion column).

### 2464350004220: thermal-printer font (back, header + column)
- **Header:** the `g` of this font looks like `s` or `3`, so `100 g` wasn't recognized and the basis is unknown. Saturated fat 4.8 then can't be decided (between 3 and 6).
- **Sodium:** RapidOCR missed `668`; Tesseract's fallback took `267` from the **40 g portion column** (267 × 2.5 = 668).
- **Fix:** §7.3 (header spellings, column discipline), §7.4.

### 2525110001719: last row shifted (back, row matching)
- **Read:** sodium 0 (true 736).
- **Why:** the bottom row's numbers sit below the label's line. "Sódio (mg)" was grouped with fibre's `0 0 0%`, leaving an **orphan row** `736 368 18%` below it.
- **Fix:** §7.1 (order alignment).

### 7895000485178: curved cup, last row shifted (back, row matching)
- **Read:** sodium 0 (true 8668).
- **Why:** same as above. The labels drift up toward the left; sodium took fibre's `0 0 0`, with the orphan row `8.668 1.300 65` below.
- **Fix:** §7.1.

### 7896022207915: the label is wrong, plus the basis wasn't read
- **Label:** saturated fat 12, which is total fat's value; the photo says 5.2, as the app read.
- **App:** answered inconclusive because the `100 g` header is cut off at the top edge. With an unknown basis, saturated fat 5.2 and sodium 320 sit between the liquid and solid limits. Yet "Porção: 30 g" and the `30 g` column say solid.
- **With the label fixed:** the truth is compliant.
- **Fix:** label; §7.4.

### 7891079000434: old format labeled as new (label wrong; app right)
- **Why:** an old-format table ("Quantidade por porção", "2000 kcal ou 8400 kJ", a product-site image of 392 × 478 px) labeled `ok` with per-portion values. The truth should be old_format → inconclusive, which is what the app answered.
- **But:** the app's reason, no_table, is wrong: the tiny white-on-red image has fewer than `MIN_TEXT_BLOBS` (40) character blobs, so no OCR ran.

### 7898971538010: old format labeled as new (label wrong; app right)
- **Why:** "Quantidade por Porção", 50 g portion, labeled `ok` with the per-portion values. The app answered old_format → inconclusive, which is right.

---

## 5. Back errors on products outside the e2e set (13)

These products have a back label but no usable front label, so they only show up in the back metrics.

| Product | Read → true | Cause |
| :--- | :--- | :--- |
| 7891000120521 | basis 100 g → 100 ml | **Truth wrong.** No basis in the label, so the truth uses Open Food Facts' package quantity (ml; an ice cream, "1 bola"); the table says `100 g`. |
| 7898101972530 | basis 100 g → 100 ml | **Truth wrong.** Same: an ice cream. |
| 7892300002791 | basis 100 g → 100 ml | **Truth wrong.** The table says `100 g` (15 g portion). |
| 7896327517702 | basis 100 ml → 100 g | Ambiguous: a chocolate powder whose table says `100 ml**` (prepared drink). IN 75 judges reconstituted products as prepared, so the app may be right. |
| 7898919916658 | saturated fat 14 → 2.5; sodium 3.8 → 1700 | Row matching (crumpled): "Gorduras totais Gorduras saturadas" grouped in one row; sodium took fibre's row, with the orphan `1700 340` below. |
| 0192505414373 | added sugars 70 → 0 | Row matching: two labels in one row (`Açúcares totais (g) Açúcares adicionados (g) 70 0 17 0 0`). |
| 7899970402838 | saturated fat 0 → 18 | Row matching after a 270° turn: `Gorduras saturadas (g) Gorduras trans (g) 0 0 0`. |
| 7898215150480 | saturated fat 75 → 4.7 | A 298 px crop: commas lost (`7,5`→`75`, `4,7`→`47`) plus a row shift. The energy (336 kcal, so at most 37 g of fat) rules out 75. |
| 7896058257656 | added sugars 3.5 → 30 | Side-by-side table: the left header was read as one token `100g30g%VD` (not a header), so the right half's `100g` column was used. |
| 7895000352371 | sodium 100 → 39 000 | The header `100g` was grouped into the sodium row and taken as the value. (Salt: Annex XVI forbidden anyway.) |
| 7896025805637 | sodium 99 → 999 | Linear text, a digit dropped: `Sódio 99mg (120mg,6%)`. The portion value and %VD contradict it (120 mg ÷ 2000 = 6%, a ~12 g portion, so ~1000 mg/100 g). |
| 0611269991000 | added sugars 1 → 11 | Linear text with glare: `11 g (28 g` read as `1 g (28 g`. The 250 ml portion makes 28 g ≈ 11.2 g/100 ml. |
| 9002490214166 | saturated fat 8 → 0 | An energy-drink can with glare: RapidOCR read 5 lines, and Tesseract's fallback produced the junk `saturados 8`. |

---

## 6. Front: misses and unknowns

- **False negatives (2):** 7898933095056 (`SÓO1O`) and 7896024762016 (glare), both in §3.
- **Unknown (6):** a panel was detected but its nutrients weren't read, so the verdict is inconclusive.

| Product | Text read in the panel | Cause |
| :--- | :--- | :--- |
| 7898215152019 | `ALTOEM`, `Do` | Nutrient not read |
| 7898633330853 | `SALTC`, `ACU`, `ADICIC` | Truncated words below the 0.7 similarity |
| 7896111423820 | `PALTOEM`, `AC`, `ONADO` | Truncated words |
| 7896181711827 | nothing | Vertical panel (133 × 394 px) |
| 7896025805637 | nothing | Tiny panel (62 × 20 px) |
| 7896066301327 | nothing | False box (score 0.17, 721 × 54 px); the label has no warning |

---

## 7. Proposed solutions, by impact

### 7.1 Match numbers to rows using geometry, not the grouping tolerance (~10 cases)
Cases: 7898571520156, 5010477352897, 7898311851410, 7891164005832, 2525110001719, 7895000485178, 7898919916658, 0192505414373, 7899970402838, 7898215150480.
- **Follow the label's own slope.** Keep RapidOCR's four-corner boxes (`words_from_lines` currently reduces them to upright rectangles) and extend each label's baseline to the 100 g column.
- **Pick vertically.** Within the column, take the number closest **vertically**, not in x (`value_in_column`).
- **Use row order.** Pair label rows with number rows top to bottom. A row of numbers with no label right below a label means a one-row shift (the orphan rows `736 368 18%`, `8.668 1.300 65`, `1700 340`).
- **Two labels, one row.** A row holding two nutrient labels (`Gorduras totais Gorduras saturadas (g) 14 2,8`) means two rows were merged: split its numbers between them in order, or answer unknown, instead of taking the first number.
- **Tolerance.** Group with the number boxes' height, not the label boxes' (which are ~1.7× taller).

### 7.2 Consistency checks, then re-read or answer unknown (~9 cases)
Cases: 7891097013003, 7898571520156, 5010477352897, 7898311851410, 7895000528448, 2464350004220, 7896025805637, 0611269991000, 7898215150480.

All classical, on numbers already read.
- **Row triple.** 100 g value × portion/100 ≈ portion value, and %VD ≈ portion value ÷ the nutrient's daily value: saturated fat 20 g, sodium 2000 mg, added sugars 50 g, total fat 65 g, carbohydrates 300 g, protein 50 g, fibre 25 g. Because the daily values differ, the %VD also says **which nutrient** a row's numbers belong to.
- **Energy bound.** Fat ≤ kcal/9 and sugars ≤ kcal/4, per 100 g/ml.
- **Recover the 100 g value.** When it's unread but the portion value is read: portion value × 100/portion size.
- **On a contradiction.** Re-read with the medium model, or answer unknown (inconclusive) rather than decide on a contradicted value.

### 7.3 Column discipline (4 cases)
Cases: 7895000528448, 2464350004220, 7896058257656, 7895000352371.
- `first_right` shouldn't take a number lying under another header (portion, %VD) when the headers were read.
- The header token itself is never a value.
- Split merged header tokens (`100g30g%VD`).
- A value can't come after the row's next label.
- Header spellings of thermal-printer fonts: `100 s`, `1003`.

### 7.4 Basis when the header isn't read (2 cases, plus 6 compliant → inconclusive)
Cases: 7896022207915, 2464350004220.
- Fall back to "Porção: 30 g/ml" or the portion column's header (`30 g`, `200 ml`).
- On the 24 compliant → inconclusive cases, 6 are an unread basis.

### 7.5 Scope rules (4 cases)
- **Spanish-only tables** (no Portuguese words; 8445290480538) → not RDC 429 → inconclusive.
- **"EXCESO EN" octagons on the front** (7891000368572) → inconclusive (imported product).
- **Per-portion-only linear declarations** ("(x g, y%)" with no per-100 header; 7896311768332) → convert with the portion size, or abstain.
- **Oils** (5890500030011): ≥ ~90 g fat, 0 carbohydrates, 0 protein → Annex XVI forbidden, no warning required.
- **Further, from the ingredient list on the back photo:** a single basic ingredient ("Amendoim torrado", 7895000528448) → conditional exemption → inconclusive.

### 7.6 Front (2 FN and 6 unknown)
- **Look-alike digits.** Normalize digits to letters before matching (`1`→i, `0`→o): 7898933095056.
- **Prefixes inside a panel.** Inside a detected panel the only words are nutrients, so a prefix of ≥ 3 letters of a key word (`ACU`, `ADICIC`, `ONADO`) can count: 3 unknowns.
- **Rotation.** Retry tall crops (h > 1.5 w) rotated 90°: 7896181711827.
- **False boxes.** Ignore boxes with no text and a low score (< 0.25), instead of making every nutrient unknown: 7896066301327.
- **Unreadable vs absent.** Count the panel's black nutrient cells (a fixed design), so an unread cell gives "unknown", not "absent": 7896024762016.
- **Warning off the front photo** (7897685914516): run the panel detector on the back photo too. A panel there means "warning on another face": inconclusive, not missing.

### 7.7 Tesseract fallback on near-empty readings (1 case)
- When RapidOCR reads almost nothing (9002490214166: 5 lines), Tesseract's values on the same crop are mostly noise: require agreement, or answer unknown.

---

## 8. Label and ground-truth issues (to decide; nothing was edited)

| Product | Label now | Suggested | Evidence |
| :--- | :--- | :--- | :--- |
| 7896625211005 | saturated fat 4 | **14** | "14 g (4,3 g, 22%)", 30 g portion |
| 7896022207915 | saturated fat 12 | **5.2** | 12 is total fat |
| 7897685914516 | added sugars blank | **50** | legible |
| 7891164005832 | saturated fat blank | **12** | partly legible |
| 7891079000434 | `ok` | **old_format** | "Quantidade por porção", "2000 kcal ou 8400 kJ" |
| 7898971538010 | `ok` | **old_format** | "Quantidade por Porção" |
| 5890500030011 | `no_table` | **ok** + values (0 / 14 / 0, 100 ml) | readable table |
| 7898933095056 | `no_table` | **ok** + values (0 / 10.5 / 801, 100 g) | readable table |
| 7898571520156, 7898311851410 | basis blank | **100g** | header legible |

Systemic issues:
- **Empty basis falls back to OFF's package quantity.** That's wrong for products sold by volume but tabled per 100 g (ice creams: 7891000120521, 7898101972530; also 7892300002791). Of the 51 inconclusive → compliant cases, **25** are a label without a basis, 15 a `no_table` label, 9 a blank value and 3 an old-format label.
- **44 back photos labeled `no_table` were read as tables,** with all three values on 28. The two checked were readable tables: worth a relabeling pass.

### Re-classification of the 20 verdict errors

| | FP | FN | Total |
| :--- | ---: | ---: | ---: |
| App error, back | 5 | 4 | 9 |
| App error, front | 2 | 0 | 2 |
| Scope (Spanish table, per-portion format, exempt oil) | 3 | 0 | 3 |
| Label or ground truth wrong / incomplete (app right or arguably right) | 3 | 3 | 6 |

Several app errors also have a label issue (missing basis or blank value); they are counted as app errors above.

---

## 9. Reproducing

```bash
# Fresh run (writes a new OCR cache; the target cache dir must not hold old entries)
pixi run python data/evaluation/e2e_dev_20261115_fresh/scripts/run_dev.py \
    data/cache/<new_cache_dir> <out_dir>/dev_run.jsonl 8
# Metrics and case lists
pixi run python data/evaluation/e2e_dev_20261115_fresh/scripts/analyze.py <out_dir>
pixi run python data/evaluation/e2e_dev_20261115_fresh/scripts/cases.py <out_dir>
# Debug images of some products, replayed from this run's cache (no OCR)
pixi run python data/evaluation/e2e_dev_20261115_fresh/scripts/debug_cases.py <dir> <code> ...
```

`cases.py` reads `dev_run.pkl` (written by `analyze.py`) from the directory it is given. `debug_cases.py` and `lines.py` read the cache at `data/cache/back_ocr_fresh_20261115/`. A change after the OCR (parse, classify, combine) can be scored on dev from that cache in about a minute, by pointing `OcrCache`'s directory at it (the default, `data/cache/back_ocr/`, holds the older cache).
