# Marathi OCR Data Factory

**One command in. A quality-gated Marathi OCR validation pack out.**

Mine real public documents → crop hard and ordinary Marathi text → OCR pre-label →
automated QA → export an **80% hard / 20% normal** dataset. No search query to type.
No review UI required.

```bash
# Default profile pack (80 hard + 20 normal)
.venv/bin/python3.12 -m scripts.run_pipeline --max-docs 8

# Request N high-quality samples — keeps mining until the quota is filled
.venv/bin/python3.12 -m scripts.run_pipeline --target-count 1000
```

---

## What you get

| Output | Location |
|--------|----------|
| Validation package | `data/export/marathi_ocr_validation_100/` |
| Images | `…/images/0001.png` … |
| Labels | `…/validation.jsonl` (`image_filename`, `expected_text`, `issue_type`) |
| Provenance | `…/metadata.json` |
| QA report | `…/validation_report.json` |

**Hard lane** — refs, Marathi numerals, dense punctuation, OCR-hard script (LLM vision confirms).  
**Normal lane** — ordinary Marathi prose (not “easy OCR”).  
Selection is **quality → diversity → quota**. Shortfalls are reported; weak samples are never padded in.

---

## Quick start

```bash
git clone <your-repo-url> Marathi_OCR
cd Marathi_OCR

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional: LLM for discover-rank / vision QA
# brew install tesseract && install `mar` traineddata

.venv/bin/python3.12 -m scripts.run_pipeline --target-count 100
```

Open the package under `data/export/marathi_ocr_validation_100/`.

---

## Requesting N samples (`--target-count`)

Pass **`--target-count N`** (alias: **`--samples`**) to keep discovering / extracting /
validating until **N high-quality accepted** samples exist.

| Flag | Behavior |
|------|----------|
| `--target-count 100` | 80 hard + 20 normal (profile ratio) |
| `--target-count 1000` | 800 hard + 200 normal |
| `--hard-count` / `--normal-count` | Override absolute lane sizes (or with `--target-count`, fix one lane and fill the other) |
| `--allow-shortfall` | Draft export if sources run out (never pads weak samples). **Off by default when a target is set** |
| `--max-rounds` | Safety cap on mine loops (default scales with N) |
| `--max-docs` / `--docs-growth` / `--max-docs-cap` | Per-round discover budget, growth, and cumulative source cap |

```bash
# 1000 samples at the default 80/20 hard/normal ratio
.venv/bin/python3.12 -m scripts.run_pipeline --target-count 1000

# Same thing via alias
.venv/bin/python3.12 -m scripts.run_pipeline --samples 1000

# Explicit lane overrides
.venv/bin/python3.12 -m scripts.run_pipeline --hard-count 800 --normal-count 200
```

**How ratio scales:** profile `hard_count` / `normal_count` (default **80 / 20**) are treated as a
ratio. For target \(N\):

\[
\text{hard} = \mathrm{round}(N \times 80/100),\quad \text{normal} = N - \text{hard}
\]

So 100 → 80+20, 1000 → 800+200. Failures that miss quality gates are discarded; the
orchestrator mines more documents until both lanes are filled, sources are exhausted, or
`--max-docs-cap` / `--max-rounds` is hit (non-zero exit, no fake padding).

---

## Pipeline

```text
discover → render → extract → score → dedup → select
       → ocr → validate → auto_accept ↻ (until target) → export
```

| Stage | Script | Does |
|-------|--------|------|
| Discover | `scripts.discover` | Portals + built-in web seeds → PDFs (skips already-ingested URLs) |
| Render | `scripts.render` | Pages @ 300 DPI |
| Extract | `scripts.extract` | Dual-lane crops @ 600 DPI (text + visual bands) |
| Score | `scripts.score` | `hard` / `normal` / reject |
| Dedup | `scripts.dedup_candidates` | Image / bbox / phash before OCR |
| Select | `scripts.select_validation` | Source-diverse hard/normal quotas |
| OCR | `scripts.ocr` | Pluggable pre-label (default Tesseract-mar) |
| Validate | `scripts.validate` | Rules + optional Kimi vision flag |
| Accept | `scripts.auto_accept` | PDF text → else OCR as label; LLM complexity end-gate |
| Export | `scripts.export` | Final package under `data/export/` |

Orchestrator: `python -m scripts.run_pipeline`.

Useful flags: `--skip-discover` · `--skip-llm-validate` · `--allow-shortfall` · `--with-review` (optional Streamlit).

---

## Repository layout

```text
Marathi_OCR/
├── configs/           # config.yaml — profiles, seeds, paths
├── scripts/           # CLI stages (run these)
├── discovery/         # portal + DDG crawl
├── extraction/        # region mining + 600 DPI crop
├── scoring/           # complexity / ordinary-prose rules
├── ocr/               # tesseract | paddleocr | surya
├── validation/        # numerals, dups, image QA
├── export/            # package writers
├── llm/               # optional Kimi client
├── pipeline/          # config, IO, hashing, profiles, quotas
├── review/            # optional Streamlit UI
├── benchmark/         # bakeoff metrics code
├── data/              # ALL runtime data (gitignored)
├── .env.example
└── requirements.txt
```

**Rule:** code and config are tracked. **`data/`, `.env`, caches, PDFs, images, JSONL are not.**

---

## How LLMs are used (optional)

Client: `llm/client.py` · credentials from `.env` (never committed).

| Use | When | Writes GT? |
|-----|------|------------|
| Discover assist | Rank / propose PDF URLs | No |
| Vision QA | Flag image↔text mismatches + complexity lane | No |

Labels come from **PDF text layer**, else **OCR**. Pipeline still runs if the LLM is down (`--skip-llm-validate` or empty key).

```bash
# .env
LLM_ENABLED=true
LLM_API_URL=https://grid.ai.juspay.net/v1/chat/completions
LLM_API_KEY=...
LLM_MODEL=kimi-k3
```

---

## Configuration

`configs/config.yaml`

- **Profiles** — `marathi_hard_validation` (no ASCII digits) vs `general_marathi`
- **Paths** — all under `data/`
- **Seeds** — `discover_search_seeds` (no CLI `--query`)
- **OCR** — `ocr_backend: tesseract` (bakeoff can change this)
- **Export** — `validation_export_dir: data/export/marathi_ocr_validation_100`
- **Mining** — `pipeline_max_rounds`, `pipeline_docs_per_round`, `pipeline_docs_growth`, `pipeline_max_docs_cap`

Bakeoff:

```bash
.venv/bin/python3.12 -m scripts.benchmark --backends tesseract,paddleocr,surya
```

---

## Design rules (locked)

1. Crawl-first discovery — portals + web search; LLM ranks only.  
2. No silent ASCII→Marathi digit conversion under the hard profile.  
3. Normal ≠ low complexity — ordinary prose without hard symbols.  
4. Image-strict / text-soft dedup.  
5. Quality before quota — report `73/80`, never pad; with `--target-count`, keep mining instead.  
6. Auto pipeline by default — Streamlit is opt-in (`--with-review`).  
7. No GR keyword / `REFERENCE_PATTERNS` hardness — LLM vision is the hard end-gate.

---

## License / use

Intended for research and OCR evaluation on public government documents.
Respect source site terms and robots rules when crawling.
