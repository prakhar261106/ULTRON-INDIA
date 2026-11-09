# ULTRON Day 1 — Upload and Run Guide

Claude cannot push to GitHub. You upload the files yourself.

## A. Files to upload to GitHub (repository `prakhar261106/ULTRON-INDIA`, branch `main`)

### 1. REPLACE these 4 existing files (open the file on GitHub -> pencil icon -> paste new content -> Commit)
| Path | What changed |
|---|---|
| `.gitignore` | keeps `.gitkeep` files, ignores secrets/models/logs/data contents |
| `requirements.txt` | real dependency list |
| `README.md` | full project README |
| `engines/stock_selection/README.md` | full engine documentation |

### 2. ADD these NEW files (same folder paths as in the ZIP)
```
pytest.ini
config/__init__.py            config/settings.py        config/ultron_config.yaml
config/universe.py            config/universe.yaml
shared/__init__.py
shared/contracts/__init__.py  shared/contracts/types.py
shared/contracts/engine_output.py   shared/contracts/market_data.py
shared/data_validation/__init__.py  shared/data_validation/market_calendar.py
shared/data_validation/candle_validator.py
shared/logging/__init__.py    shared/logging/logger.py
engines/__init__.py
engines/stock_selection/__init__.py   engines/stock_selection/engine.py
engines/stock_selection/features.py   engines/stock_selection/ranking.py
engines/stock_selection/STATUS.md
engines/stock_selection/samples/sample_input_SYNTHETIC_candles_1h.csv
engines/stock_selection/samples/sample_output_SYNTHETIC.json
tests/__init__.py  tests/conftest.py  tests/synthetic_data.py
tests/test_config_logging_adapter.py  tests/test_data_validation.py
tests/test_engine_contract.py  tests/test_repo_structure.py  tests/test_stock_selection.py
notebooks/day1_foundation_stock_selection.ipynb
documentation/DAY1_UPLOAD_GUIDE.md
data/raw/.gitkeep   data/processed/.gitkeep   data/features/.gitkeep
```

### How to upload (easiest, no command line)
1. Unzip `ULTRON_Day1_deliverables.zip` on your computer.
2. On GitHub: **Add file -> Upload files**. Drag the unzipped folders (`config`, `shared`, `engines`, `tests`, `notebooks`, `documentation`) and the file `pytest.ini`. GitHub keeps the folder paths. Commit.
3. **`.gitkeep` files are hidden** and often skipped by drag-and-drop. Create each one on GitHub instead: **Add file -> Create new file**, type `data/raw/.gitkeep` as the file name (typing `/` makes the folder), leave the content empty, commit. Repeat for `data/processed/.gitkeep` and `data/features/.gitkeep`.
4. Replace the 4 existing files (section 1). Do **not** edit any other existing file.

## B. Open the notebook in Google Colab
1. On GitHub open `notebooks/day1_foundation_stock_selection.ipynb`.
2. Click the **Open in Colab** badge, or go to https://colab.research.google.com -> File -> Open notebook -> **GitHub** tab -> paste `prakhar261106/ULTRON-INDIA` -> choose the notebook.
3. (Alternative, if the files are not on GitHub yet) Open Colab, upload the notebook from the ZIP, set `SETUP_MODE = "zip"` in the first code cell, and upload the ZIP when asked.

## C. Which cells to run, in order
Run **Cell 1 to Cell 13 from top to bottom** (Runtime -> Run all works). The optional Google Drive cell near the end is commented out.

| Cell | What it does | What you should see |
|---|---|---|
| 1 Environment | gets the code | `OK - Day 1 files found.` |
| 2 Dependencies | pip install | no error |
| 3 Imports | loads modules | python/pandas versions |
| 4 Config | prints settings | `Data provider: None` |
| 5 Universe | loads NIFTY 50 + BANK NIFTY | `58 unique symbols (5 duplicates removed)` (count can change with date) |
| 6 Adapter | refusal + synthetic demo data | `Expected refusal -> No market-data provider...` |
| 7 Validation | good vs damaged data | `GOOD ... valid: True`, `DAMAGED ... valid: False` |
| 8 Features | numbers for one stock | table of 9 features |
| 9 Ranking | runs the engine | `eligible: 56 | excluded: 2 | shortlisted: 10` |
| 10 Shortlist | top 10 + contract JSON | 10 rows, direction NEUTRAL |
| 11 Exclusions | why 2 stocks were excluded | DEMO-ITC, DEMO-TCS |
| 12 Tests | runs pytest | `86 passed` (ZIP mode: `85 passed, 1 skipped` - normal), `ALL TESTS PASSED` |
| 13 Save | writes 4 result files | paths under `data/processed/stock_selection/` |

The shortlist in this notebook is **SYNTHETIC demo output**, not real stock picks.

## D. Run the tests outside Colab (optional)
```
pip install -r requirements.txt
python -m pytest
```
