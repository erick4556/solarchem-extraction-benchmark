# SolarChem extraction benchmark

Benchmark for extracting **tables, figures, and their context** from SolarChem
scientific PDFs. The current release scores **tables and table-context**; figure
extraction is planned on the same schema and evaluation CLI.

Automatic generation of a **table and table-context ground truth** for the
SolarChem PDF corpus, in a form that can be compared directly against the
output of any extraction tool (pdfplumber, Camelot, Unstructured, TATR, Docling,
GROBID, LightOnOCR, Unlimited-OCR, Ollama VLMs).

The generation strategy follows the GAP-KGE project: an OCR engine transcribes
each page, the emitted tables are flattened into a rectangular grid, and the
narrative context is resolved with positional heuristics. Two differences are
deliberate:

* **tables and their context live in the same schema** and the default output is
  a **single corpus JSON file**;
* the ground truth holds **only fields that are scored**. Hashes, engine names,
  timestamps and review state belong to the run manifest, so two ground-truth
  files can be diffed without stripping metadata first.

## Layout

This repository is `code/` plus `data/` (except the PDF corpus):

```
.
├── README.md             # this file (GitHub landing page)
├── code/                 # package, scripts, tests
└── data/
    ├── documents/        # source PDF corpus (gitignored; not in the repo)
    ├── ground_truth/     # generated ground truth JSON (per engine)
    ├── predictions/      # tool outputs (not GT)
    └── intermediate/
        └── ocr_cache/
            ├── lighton_ocr/      # raw page transcriptions (LightOn)
            └── unlimited_ocr/    # raw page transcriptions (Unlimited)
```

`data/documents/` is excluded from git. Ground truth and predictions are tracked.
All `python scripts/…` commands below run from `code/`.
No path is hardcoded. The data root is resolved from `--data-root`, then from
`$SOLARCHEM_DATA_ROOT`, then by looking for `data/` next to or inside the
repository.

## Install

On the GPU server, install the base package without touching the existing
CUDA-enabled PyTorch:

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then add the OCR backend. **LightOnOCR-2 needs transformers ≥ 5.0** (the
classes are not in 4.57.6). Do not reinstall `torch`:

```bash
pip install -e ".[lighton_ocr]" --no-deps
pip install "transformers>=5.0" accelerate
```

Unlimited-OCR's model card pins 4.57.1; on transformers 5 the adapter shims
the removed `is_torch_fx_available` import:

```bash
pip install -e ".[unlimited_ocr]" --no-deps
pip install einops addict easydict
```

## Usage

Smoke test on one document and the first few pages:

```bash
python scripts/generate_ground_truth.py --limit 1 --max-pages 8 --log-level DEBUG
```

One specific PDF with an explicit identifier:

```bash
python scripts/generate_ground_truth.py \
  --pdf ../data/documents/1-s2.0-S0021979721022451-main.pdf \
  --document-id solarchem_doc_001
```

The whole corpus (default: single-file output):

```bash
python scripts/generate_ground_truth.py
```

Custom single-file path:

```bash
python scripts/generate_ground_truth.py --output ../data/ground_truth/ground_truth_lighton_ocr.json
```

By default (no `--output`) the merged file is named after the engine:

* LightOn → `data/ground_truth/ground_truth_lighton_ocr.json`
* Unlimited → `data/ground_truth/ground_truth_unlimited_ocr.json`

Optional per-document mode:

```bash
python scripts/generate_ground_truth.py --per-document
```

Validate and summarise what was produced:

```bash
python scripts/validate_ground_truth.py
python scripts/validate_ground_truth.py --dump-schema schemas/ground_truth.schema.json
```

Transcriptions are cached per document and engine, so re-running after a parser
change costs no GPU time. The ground-truth JSON itself also resumes: documents
already present in `ground_truth_<engine>.json` (or as a per-document file) are
skipped, and each successful PDF is appended to the merged file immediately, so
a crash mid-corpus does not throw away the work already done. PDFs from which
no table is recovered are omitted from the JSON (the OCR cache is still kept).
Use `--overwrite` to regenerate documents already in the output and `--force-ocr`
to ignore the OCR cache.

## Output

By default a single file is produced, named after the OCR engine:

* `data/ground_truth/ground_truth_lighton_ocr.json` (default `--engine lighton_ocr`)
* `data/ground_truth/ground_truth_unlimited_ocr.json` (`--engine unlimited_ocr`)

Optional with `--per-document`:

* `data/ground_truth/<document_id>.json`

See [`code/examples/solarchem_example.json`](code/examples/solarchem_example.json) for a
full example of one document entry produced by the pipeline.

```json
{
  "document_id": "solarchem_example",
  "source_pdf": "example.pdf",
  "title": "Construction of NiO/g-C3N4 p-n heterojunctions for enhanced photocatalytic CO2 reduction",
  "num_tables": 1,
  "tables": [
    {
      "table_id": "solarchem_example_table_01",
      "table_label": "Table 1",
      "page": 1,
      "caption": "Table 1: Photocatalytic CO2 reduction activity over TiO2-based catalysts.",
      "columns": ["Catalyst", "Production rate (umol g^-1 h^-1)_CH4", "Selectivity (%)"],
      "rows": [["TiO2", "12.5 ± 0.8", 80.1], ["Pt/TiO2", "21.7 ± 1.2", ">99"]],
      "context": {
        "section_title": "3. Results and discussion",
        "mentions": [{ "page": 2, "text": "As summarised in Table 1, ..." }]
      }
    }
  ]
}
```

### What each field is for

| Field | Evaluates |
| --- | --- |
| `title` | the article title, read from the PDF metadata |
| `page`, `table_label` | table detection and identification |
| `caption` | caption extraction, verbatim as printed |
| `columns`, `rows` | structure and content: column headers and cell accuracy |
| `context.section_title` | which section of the article the table belongs to |
| `context.mentions` | retrieval of in-text references |

The table's own title is not stored: it is the caption minus its label, so
`caption_title()` derives it at read time and there is no second copy to keep
in sync.

```python
from solarchem_benchmark.gt import caption_title

caption_title("Table 1 Optical properties of UiO-66.")  # "Optical properties of UiO-66."
```

## Design decisions

**Tables are flattened, not modelled cell by cell.** Merged cells are expanded
so every grid position holds the value of the region covering it, and
multi-level headers are collapsed into one label per column, joining distinct
levels with `_`. This is GAP-KGE's strategy, which keeps both benchmarks
comparable.

**Composite cell values are preserved verbatim.** `12.5 ± 0.8`, `<0.01`,
`1.2 × 10^-3` and `ND` stay as strings. Splitting them into value, uncertainty
and operator is the evaluation layer's job; doing it in the ground truth would
bake one interpretation into the reference.

**Empty is not the same as not detected.** Blank cells and dash placeholders
become `-`. Textual markers such as `ND` or `N/A` are kept, because "not
detected" is itself an experimental result.

**Case is never folded.** `Co` is cobalt and `CO` is carbon monoxide.

**A caption may be printed as two blocks.** Elsevier and others typeset the
label on one line and the caption text on the next, so a block holding nothing
but `Table 1` absorbs the block below it. Without this the caption collapses to
the bare label and its text is lost.

**Nothing derivable is stored twice.** A hand-curated reference is edited by
people, and any field computable from another is a field that will eventually
disagree with it. The table's title is the clearest case: it is the caption
without its label, so only the caption is kept.

**The article title comes from the PDF, not from the OCR.** On a first page the
title is typographically indistinguishable from the running head, the journal
name and the author list, whereas the `/Title` metadata entry is exact when
present. It is available for 70% of the corpus; the rest keep an empty title,
because for a reference an absent value beats a guessed one. Publisher
artefacts left in that entry -- `ja304075b 1..6`, `No Job Name`, source
filenames -- are rejected.

**Sub- and superscripts stay attached to their base.** `CH<sub>4</sub>` reads
`CH4` and `g<sup>-1</sup>` reads `g^-1`. Reading such a cell by joining its text
nodes with a space, which is the obvious implementation, splits formulas and
units into `CH 4` and `g -1`.

**Every text field is normalised, not just the cells.** An engine writes the
same species differently depending on where it lands: `H$_2$BDC` in a caption,
`H $_2$ BDC` in a heading, `H<sub>2</sub>BDC` in a cell. Captions, section
titles and mentions therefore go through the same normaliser as cells, which
expands LaTeX math as well as Unicode scripts. Without it a table cannot be
linked to the prose that describes it, and a formatting difference would score
as an extraction error.

A script binds to the base before it, so the space an engine inserts in front
of it is always dropped; the space after it is dropped only when the formula
visibly continues. `H $_2$ BDC` becomes `H2BDC` while `CO $_2$ reduction`
stays two words.

**Context is limited to location and reference.** A table's context is the
section it sits under and the paragraphs that name it explicitly. Neighbouring
paragraphs and domain entity lists were deliberately dropped: entity extraction
belongs to the later semantic phase, where it is done with a model and scored,
not baked into the reference.

**The OCR engine is a parameter, not a constant.** Which engine produced a draft
is a property of the run. Keeping the choice configurable is what allows the
later engine-comparison experiment to avoid a generator that quietly favours
itself.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers normalisation, flattening, entity extraction and end-to-end
assembly using synthetic transcriptions, so it runs without a GPU or model
weights.

## Evaluate automatic GT against the Gold pilot

The Gold pilot has **10 documents**. Notation follows the same canonical form
as the extractor (`m^2`, header `_` joins). `section_title` in Gold is the
discussing section (semantic); silver uses the heading above the table
(positional). The evaluator only scores reference overlap and lists missing
reference PDFs.

The report `summary` is grouped by field — `detection`, `columns`, `cells`,
and, with `--metrics all`, `caption`, `section`, `mentions`. Each field
exposes **accuracy**, **precision**, **recall** and **f1**. Accuracy is
exact string or header-list agreement (Jaccard for detection and mentions).
P/R/F1 are token overlap for caption, columns and section; set overlap for
mentions; table-level for detection. On cells the four numbers coincide:
the grid is compared position by position. `--metrics structure` omits the
context fields so empty captions are not ranked.

1. Copy `data/ground_truth/gold/` to the server (next to the silver JSON files).
2. If the 10 Gold PDFs are not already inside silver, generate them:
   ```bash
   python scripts/generate_ground_truth.py \
     --engine lighton_ocr \
     --pdf ../data/documents/Novel_Ti-KIT-6_material_for_the_photocatalytic_reduction_of_carbon_dioxide_to_methane.pdf
   ```
   (resume appends into `ground_truth_lighton_ocr.json`).
3. Score:
   ```bash
   python scripts/evaluate_against_gold.py \
     --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
     --engine lighton_ocr \
     --report ../data/ground_truth/gold/eval_report_lighton_ocr.json
   ```

See [`code/docs/gold_pilot.md`](code/docs/gold_pilot.md).

## Phase 5 — native PDF table extractors

These tools read the PDF text layer (no GPU). They write the **same schema** as
the ground truth, with **structure only**: `columns` / `rows` are filled;
`caption` is set only when a `Table N` spanner sits inside the grid;
`section_title` and mentions stay empty. Score them with `--metrics structure`
so empty context fields are not treated as a ranking.

Do **not** score LightOn against `ground_truth_lighton_ocr_302.json` (it produced
that file). Score LightOn against Gold only.

Install the CPU extras (pick one or all). Camelot lattice also needs
**Ghostscript** on `PATH`.

```bash
pip install -e ".[pdfplumber]"     # start here
pip install -e ".[pymupdf]"
pip install -e ".[camelot]"        # + Ghostscript for lattice
# or: pip install -e ".[native_pdf]"
```

By default only the PDFs listed in the frozen working silver
(`data/ground_truth/ground_truth_lighton_ocr_302.json`, 302 documents) are
processed. Output goes to `data/predictions/<tool>.json`, not into the GT
files. Documents already present are skipped (resume). Documents with 0
recovered tables are **kept** (`num_tables: 0`) so the prediction file has the
same 302 keys as silver; the evaluator counts those as detection misses.
Cell/column scores are only computed on matched tables.

```bash
# smoke test
python scripts/extract_native_tables.py --tool pdfplumber --limit 1 --max-pages 8

# working set (302 PDFs)
python scripts/extract_native_tables.py --tool pdfplumber
python scripts/extract_native_tables.py --tool pymupdf
python scripts/extract_native_tables.py --tool camelot_lattice
python scripts/extract_native_tables.py --tool camelot_stream
```

Evaluate structure against Gold-10 (paper numbers) and against the working
silver (ranking on the 302-doc bank):

```bash
python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --tool pdfplumber \
  --metrics structure \
  --report ../data/predictions/eval_pdfplumber_gold10.json

python scripts/evaluate_against_gold.py \
  --reference ../data/ground_truth/ground_truth_lighton_ocr_302.json \
  --tool pdfplumber \
  --metrics structure \
  --report ../data/predictions/eval_pdfplumber_silver302.json
```

`--tool` picks `data/predictions/<tool>.json`. Pass `--prediction` to score any
other file.

## Phase 6 — layout / end-to-end extractors

Docling, Table Transformer (TATR via gmft) and Unstructured write the same
schema as Phase 5. Score them with `--metrics structure`. Caption/section stay
empty unless the library put a `Table N` spanner in the grid. Unstructured uses
`strategy=hi_res` (slow, first run downloads layout models). Docling has OCR
off so it is a layout tool, not a second OCR engine.

On the JupyterHub stack, conda `numexpr`/`bottleneck`/`scikit-learn` may be
built for NumPy 1 while Docling pulls NumPy 2. The extractor skips those pandas
extras, restores ``numpy.core.numeric.ComplexWarning``, and stubs sklearn so
transformers can load the layout detector (Docling does not need sklearn).
Do not reinstall torch. Optional: `pip install --user --upgrade scikit-learn`.
The first Docling convert is slow (layout models); let it finish.

```bash
pip install -e ".[docling]"
python scripts/extract_native_tables.py --tool docling

pip install -e ".[tatr]" --no-deps
pip install gmft
python scripts/extract_native_tables.py --tool tatr

pip install -e ".[unstructured]"
# JupyterHub: conda SciPy/sklearn may be NumPy 1 while this runtime is NumPy 2.
# Do not reinstall torch.
pip install --user --upgrade scipy scikit-learn
python scripts/extract_native_tables.py --tool unstructured
```

Same evaluation as Phase 5, changing `--tool`:

```bash
python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --tool docling --metrics structure \
  --report ../data/predictions/eval_docling_gold10.json

python scripts/evaluate_against_gold.py \
  --reference ../data/ground_truth/ground_truth_lighton_ocr_302.json \
  --tool docling --metrics structure \
  --report ../data/predictions/eval_docling_silver302.json
```

## Phase 7 — GROBID and Ollama VLMs (tables + context)

GROBID is the classical scholarly parser (CPU, TEI XML): cell grids when the
table model emits ``<row>/<cell>``, plus caption, nearest section heading, and
in-text table references. Same CLI, schema and eval as Docling. Needs a running
GROBID server (first start downloads models and is slow):

```bash
docker run --rm -p 8070:8070 grobid/grobid:0.8.2
```

Same extract + two evals as the other tools (`--metrics all` because GROBID
fills context):

```bash
python scripts/extract_native_tables.py --tool grobid
# → data/predictions/grobid.json

python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --tool grobid --metrics all \
  --report ../data/predictions/eval_grobid_gold10.json

python scripts/evaluate_against_gold.py \
  --reference ../data/ground_truth/ground_truth_lighton_ocr_302.json \
  --tool grobid --metrics all \
  --report ../data/predictions/eval_grobid_silver302.json
```

Override the URL with `--grobid-host` or `$GROBID_HOST` (default
`http://127.0.0.1:8070`). Resume is default.

Local vision-language models extract **grids and caption / section / mentions**
through a running Ollama server (`ollama serve`). Page images use the same
render as LightOnOCR (200 DPI, longest side 1540 px). Mentions are filled in a
second text-only pass over the PDF text layer. Load **one** model at a time on the H100 MIG 47 GB slice. Start `ollama serve`
from a directory that will not disappear (`cd /tmp && ollama serve`).
If llama-server dies with ``cannot get current path``, the serve process was
started from a deleted cwd. An HTTP 200 with empty ``message.content`` on
``qwen3-vl`` is usually ``num_ctx`` smaller than the page-image tokens, or
JSON Schema structured output; the extractor retries with ``format=json``.

Default tags: `qwen3-vl:32b`, `gemma4:31b`, `mistral-small3.2:24b`.
Override with `--ollama-model` or `$OLLAMA_HOST`.

Start on Gold-10 (not the 302-doc silver) — each page is a GPU call.
Predictions still go to `data/predictions/<tool>.json`, same as TATR/Docling.
`gold10` appears only on the evaluation report.

```bash
# terminal 1
ollama serve

# terminal 2
python scripts/extract_native_tables.py \
  --tool ollama_qwen3_vl \
  --reference ../data/ground_truth/gold/ground_truth_gold_pilot.json

python scripts/extract_native_tables.py \
  --tool ollama_gemma4 \
  --reference ../data/ground_truth/gold/ground_truth_gold_pilot.json

python scripts/extract_native_tables.py \
  --tool ollama_mistral_small \
  --reference ../data/ground_truth/gold/ground_truth_gold_pilot.json
```

Smoke test: `--limit 1 --max-pages 2`. Resume is default; `--overwrite` to redo.

Evaluate with **`--metrics all`** (structure + context):

```bash
python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --tool ollama_qwen3_vl --metrics all \
  --report ../data/predictions/eval_ollama_qwen3_vl_gold10.json
```

Same pattern for `ollama_gemma4` and `ollama_mistral_small`. Structure-only
against silver-302 is optional and slow; do not score these VLMs as if they
produced the silver file.

## Status

Implemented and tested: the flattening, context and normalisation layers, the
LightOnOCR-2 adapter, the Unlimited-OCR adapter (`--engine unlimited_ocr`),
both GT CLIs (with resume), the schema, a **Gold pilot** of 10 human-verified
tables under `data/ground_truth/gold/` (see [`code/docs/gold_pilot.md`](code/docs/gold_pilot.md)),
and **Phase 5 native extractors** (pdfplumber, Camelot lattice/stream, PyMuPDF)
plus **Phase 6 layout extractors** (Docling, TATR, Unstructured), **GROBID**,
and **Phase 7 Ollama VLMs** writing to `data/predictions/`.

Unlimited-OCR uses `baidu/Unlimited-OCR` with `trust_remote_code` and requires
CUDA. Each engine writes its own silver file by default:

```bash
# deps (do not reinstall an existing CUDA torch)
pip install -e ".[unlimited_ocr]" --no-deps
pip install einops addict easydict

python scripts/generate_ground_truth.py \
  --engine unlimited_ocr \
  --overwrite \
  --pdf ../data/documents/<gold-pdf>.pdf
# → data/ground_truth/ground_truth_unlimited_ocr.json

python scripts/evaluate_against_gold.py \
  --gold ../data/ground_truth/gold/ground_truth_gold_pilot.json \
  --engine unlimited_ocr \
  --report ../data/ground_truth/gold/eval_report_unlimited_ocr.json
```

OCR caches are per engine under `data/intermediate/ocr_cache/<engine_id>/`
(e.g. `…/lighton_ocr/*.json` vs `…/unlimited_ocr/*.json`).
On an A100 the SGLang recipe's `--attention-backend fa3` is unavailable
(FlashAttention-3 needs Hopper); the Transformers adapter used here does not
require FA3.
