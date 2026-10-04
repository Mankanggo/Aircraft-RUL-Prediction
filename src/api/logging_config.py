"""JSON-lines logging to stdout for the API (does not use src.logger, which writes files at import)."""

import json
import logging
import sys
import time

LOGGER_NAME = "rul_api"

# Structured fields that may be attached via `extra=`; raw sensor payloads are never logged.
_FIELDS = ("request_id", "method", "endpoint", "status", "latency_ms", "dataset", "n_engines", "n_rows",
           "model_family", "config_version", "error_code", "detail")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _FIELDS:
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Idempotent: attaches one stdout handler to the API logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_rul_api", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._rul_api = True  # marker for idempotence
        logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
