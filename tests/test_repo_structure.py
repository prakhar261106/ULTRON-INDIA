"""Checks that the repository foundation (data folders, .gitkeep, .gitignore rules) is correct."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = ["data/raw", "data/processed", "data/features"]


@pytest.mark.parametrize("d", DATA_DIRS)
def test_data_folders_have_gitkeep(d):
    assert (ROOT / d / ".gitkeep").exists()


def test_required_foundation_files_exist():
    for f in ["README.md", "requirements.txt", ".gitignore", "pytest.ini", "config/ultron_config.yaml",
              "config/universe.yaml", "shared/contracts/engine_output.py", "shared/contracts/market_data.py",
              "shared/data_validation/candle_validator.py", "shared/logging/logger.py",
              "engines/stock_selection/engine.py"]:
        assert (ROOT / f).exists(), f


def _ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    return r.returncode == 0


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_gitignore_rules():
    for kept in ["data/raw/.gitkeep", "data/processed/.gitkeep", "data/features/.gitkeep", ".env.example",
                 "engines/stock_selection/samples/sample_output_SYNTHETIC.json", "requirements.txt"]:
        assert not _ignored(kept), f"{kept} must NOT be ignored"
    for hidden in ["data/raw/prices.csv", "data/processed/x.parquet", "data/features/f.csv", ".env",
                   ".env.local", "secrets/key.txt", "model.pt", "model.pkl", "logs/run.log", "credentials_dhan.json"]:
        assert _ignored(hidden), f"{hidden} must be ignored"
