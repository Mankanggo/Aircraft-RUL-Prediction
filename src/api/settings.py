"""Environment-based settings for the RUL API (no file writes, no working-directory assumptions)."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """
    model_dir / config_dir    frozen artifacts (RUL_MODEL_DIR / RUL_CONFIG_DIR); defaults resolve
                              relative to the project root, never to the working directory
    max_*                     request limits; exceeding them returns 413
    max_concurrent_predictions  simultaneous predictor calls per process. The frozen models use
                              4 threads each, so this bounds CPU threads to ~4 x this value.
    near_cap_margin           predictions >= cap - margin are flagged at_or_near_cap
    short_history_cycles      histories with <= this many cycles are flagged short_history
    fail_fast                 abort startup if the frozen artifacts fail verification; if False the
                              process stays up but /ready and prediction endpoints return 503
    """

    model_dir: Path
    config_dir: Path
    max_engines: int = 500
    max_cycles_per_engine: int = 1000
    max_total_rows: int = 100_000
    max_request_bytes: int = 64 * 1024 * 1024
    max_upload_bytes: int = 16 * 1024 * 1024
    max_concurrent_predictions: int = 2
    near_cap_margin: float = 5.0
    short_history_cycles: int = 30
    fail_fast: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=Path(os.environ.get("RUL_MODEL_DIR") or PROJECT_ROOT / "models" / "final").resolve(),
            config_dir=Path(os.environ.get("RUL_CONFIG_DIR") or PROJECT_ROOT / "configs").resolve(),
            max_engines=_env_int("RUL_MAX_ENGINES", cls.max_engines),
            max_cycles_per_engine=_env_int("RUL_MAX_CYCLES_PER_ENGINE", cls.max_cycles_per_engine),
            max_total_rows=_env_int("RUL_MAX_TOTAL_ROWS", cls.max_total_rows),
            max_request_bytes=_env_int("RUL_MAX_REQUEST_BYTES", cls.max_request_bytes),
            max_upload_bytes=_env_int("RUL_MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            max_concurrent_predictions=_env_int("RUL_MAX_CONCURRENT_PREDICTIONS", cls.max_concurrent_predictions),
            near_cap_margin=_env_float("RUL_NEAR_CAP_MARGIN", cls.near_cap_margin),
            short_history_cycles=_env_int("RUL_SHORT_HISTORY_CYCLES", cls.short_history_cycles),
            fail_fast=_env_bool("RUL_FAIL_FAST", cls.fail_fast),
            log_level=os.environ.get("RUL_LOG_LEVEL", cls.log_level).upper(),
        )
