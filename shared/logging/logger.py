"""Logging utilities for ULTRON.

* Console logging, plus an optional log file.
* Secret redaction: anything that looks like ``api_key=...``, ``token: ...``, ``password=...``
  is replaced by ``***REDACTED***`` before it is written anywhere.
* Safe to call ``get_logger`` many times: handlers are never duplicated.

Note: this package is named ``shared.logging``. Always import it as
``from shared.logging import get_logger`` from the repository root. (Inside this folder,
``import logging`` still means Python's standard library.)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.contracts.types import IST

_SECRET_RE = re.compile(
    r"(?i)(api[_\-]?key|access[_\-]?token|token|secret|password|passwd|authorization|client[_\-]?secret)"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Replace the value part of ``key=value`` pairs that look like secrets."""
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = redact_secrets(record.getMessage())
        record.msg, record.args = message, ()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, IST).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(
    name: str = "ultron",
    level: str | int = "INFO",
    log_dir: Optional[str | Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Return a configured logger (console + optional file in ``log_dir``)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if any(getattr(h, "_ultron", False) for h in logger.handlers):
        return logger  # already configured

    formatter: logging.Formatter = _JsonFormatter() if json_format else logging.Formatter(_TEXT_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(SecretRedactingFilter())
        handler._ultron = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def log_engine_event(logger: logging.Logger, engine: str, event: str, **fields) -> None:
    """One-line structured event, e.g. ``log_engine_event(log, 'stock_selection', 'run_done', n=10)``."""
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[%s] %s %s", engine, event, extras)
