# Gold pilot ground truth

Human-verified reference used for OCR selection and paper numbers. It is
**not** the automatic silver corpus.

## Files

| Path | Role |
| --- | --- |
| `data/ground_truth/gold/ground_truth_gold_pilot.json` | Evaluable GT (same schema as silver) |
| `data/ground_truth/gold/curation_log.json` | Provenance, conventions, exclusions |

## Conventions (no cheating)

Gold and silver share the **same notation rules** as
`normalize_scientific_text` / header flatten:

* plain ASCII (no LaTeX in the JSON)
* unit powers with caret: `m^2`, `cm^3`, `g^-1`
* multi-level headers joined with `_`
* angstrom → `A`, micro → `u`

What Gold does **not** do:

* copy silver predictions to raise scores
* retarget `section_title` to whatever silver emitted

### `section_title`

| | Gold | Silver |
| --- | --- | --- |
| Meaning | discussing section (semantic, human) | heading above table float (positional) |
| Primary metric | section P/R/F1 (tokens) | section P/R/F1 (tokens) |

Section **accuracy** (identical string) is secondary: `2 Results` vs
`2. Results` is a miss on accuracy and a hit on F1.

### Mentions

Gold: concise citing sentences. Silver: often full OCR paragraphs. Score
mention **recall** (and the rest of the mention quartet), not identical
strings.

## Pilot contents

10 documents, 10 tables (one `Table 1` each): lattice, textural, band gap,
EDS, transposed calcination, phase composition, Me/TiO₂, formic-acid yield,
HZSM-5 rates.

## Validate

```bash
python scripts/validate_ground_truth.py \
  --input ../data/ground_truth/gold/ground_truth_gold_pilot.json
```

## Evaluate silver against Gold

```bash
python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --engine lighton_ocr \
  --report ../data/ground_truth/gold/eval_report_lighton_ocr.json
```

Prefer cell **accuracy** plus caption / section **F1** and mention **recall**
for OCR selection. Column and section **accuracy** stay stricter by design.

Silver files are named per engine by default:

* `ground_truth_lighton_ocr.json`
* `ground_truth_unlimited_ocr.json`
