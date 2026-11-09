"""Universe loader: NIFTY 50 + BANK NIFTY with duplicate removal and point-in-time membership."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from shared.contracts.types import normalize_symbol

DEFAULT_UNIVERSE_PATH = Path(__file__).resolve().parent / "universe.yaml"
DEFAULT_INDICES = ("NIFTY50", "BANKNIFTY")


class UniverseMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    sector: str = "UNKNOWN"
    indices: List[str]
    added: Optional[date] = None
    removed: Optional[date] = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v):
        return normalize_symbol(v)


class Universe(BaseModel):
    """Duplicate-free list of members that were active on ``as_of``."""

    model_config = ConfigDict(extra="forbid")

    as_of: date
    members: List[UniverseMember]
    membership_history_verified: bool
    warnings: List[str] = []

    @property
    def symbols(self) -> List[str]:
        return [m.symbol for m in self.members]

    @property
    def sector_of(self) -> Dict[str, str]:
        return {m.symbol: m.sector for m in self.members}


def dedupe_symbols(symbols: Iterable[str]) -> List[str]:
    """Normalise and remove duplicates while keeping the first-seen order."""
    seen, out = set(), []
    for s in symbols:
        n = normalize_symbol(s)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _active(entry: dict, as_of: date) -> bool:
    added, removed = entry.get("added"), entry.get("removed")
    if added is not None and as_of < added:
        return False
    if removed is not None and as_of >= removed:
        return False
    return True


def load_universe(
    as_of: date | datetime | None = None,
    indices: Sequence[str] = DEFAULT_INDICES,
    path: str | Path | None = None,
) -> Universe:
    """Load the universe for ``as_of`` (default: today).

    * Symbols that appear in several indices (e.g. HDFCBANK) are returned ONCE.
    * Only entries whose membership window contains ``as_of`` are kept.
    """
    if as_of is None:
        as_of = date.today()
    elif isinstance(as_of, datetime):
        as_of = as_of.date()

    with open(Path(path) if path else DEFAULT_UNIVERSE_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    merged: Dict[str, dict] = {}
    for index in indices:
        if index not in raw:
            raise KeyError(f"Index {index!r} not found in universe file. Available: "
                           f"{[k for k in raw if k.isupper()]}")
        for entry in raw[index]:
            if not _active(entry, as_of):
                continue
            sym = normalize_symbol(entry["symbol"])
            rec = merged.setdefault(
                sym,
                {"symbol": sym, "sector": entry.get("sector", "UNKNOWN"), "indices": [],
                 "added": entry.get("added"), "removed": entry.get("removed")},
            )
            rec["indices"].append(index)

    verified = bool(raw.get("membership_history_verified", False))
    warnings: List[str] = []
    if not verified:
        warnings.append(
            "UNIVERSE_NOT_POINT_IN_TIME: membership history is incomplete; for past dates this "
            "list may contain survivorship bias. Verify against NSE Indices files."
        )
    members = [UniverseMember(**rec) for rec in merged.values()]
    return Universe(as_of=as_of, members=members, membership_history_verified=verified, warnings=warnings)
