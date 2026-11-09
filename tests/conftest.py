"""Shared pytest fixtures. All market data used in tests is SYNTHETIC (see tests/synthetic_data.py)."""
from datetime import date, datetime

import pytest

from config.settings import load_config
from shared.contracts.types import IST
from tests.synthetic_data import make_synthetic_market

AS_OF = datetime(2026, 11, 6, 15, 30, tzinfo=IST)  # Friday, after the close
END_DATE = date(2026, 11, 6)


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def as_of():
    return AS_OF


@pytest.fixture(scope="session")
def small_market():
    """3 synthetic symbols, 25 sessions of 15m candles, plus a synthetic index."""
    return make_synthetic_market(["AAA", "BBB", "CCC"], "15m", 25, END_DATE)


@pytest.fixture
def candles(small_market):
    stocks, _ = small_market
    return stocks["AAA"].copy()


def synthetic_universe_market(symbols, tf="15m", n_sessions=25, **kw):
    return make_synthetic_market(symbols, tf, n_sessions, END_DATE, **kw)
