# Data directory

All pipeline inputs/outputs live here. **Nothing under `data/` is committed**
(except this README). Folders are created automatically on first run.

```text
data/
  sources/          downloads, manifest, exclude list, discovery logs
  raw/              ingested source PDFs
  rendered/         full-page PNGs (300 DPI)
  candidates/       600 DPI crops + stage JSONL
  reviewed/         auto-accepted / reviewed labels
  final/            legacy JSONL export
  export/           packaged validation set
    marathi_ocr_validation_100/
  benchmark/        OCR bakeoff images + labels (local only)
  cache/            OCR recognition cache
```

Regenerate everything:

```bash
.venv/bin/python3.12 -m scripts.run_pipeline --max-docs 8
```

Package output: `data/export/marathi_ocr_validation_100/`
