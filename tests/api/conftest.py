import dataclasses
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.settings import PROJECT_ROOT, Settings
from src.data_schema import RAW_COLUMNS, SUBSETS
from src.pipelines.prediction_pipeline import RULPredictor

GOLDEN_DIR = PROJECT_ROOT / "tests" / "fixtures" / "api_golden"
MODEL_DIR = PROJECT_ROOT / "models" / "final"
CONFIG_DIR = PROJECT_ROOT / "configs"
FORBIDDEN_IN_RESPONSES = ("Traceback", "site-packages", "C:\\", "/Users/", "np.int64", "File \"", str(PROJECT_ROOT))


def default_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(model_dir=MODEL_DIR, config_dir=CONFIG_DIR, log_level="WARNING"), **overrides)


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app(default_settings())) as c:
        yield c


@pytest.fixture(scope="session")
def predictors():
    return {s: RULPredictor.load(s, model_dir=str(MODEL_DIR)) for s in SUBSETS}


@pytest.fixture(scope="session")
def golden():
    return {s: json.loads((GOLDEN_DIR / f"{s}.json").read_text(encoding="utf-8")) for s in SUBSETS}


def request_frame(request: dict) -> pd.DataFrame:
    """The raw DataFrame a JSON request describes (independent of the API's own conversion)."""
    rows = [{"unit_id": e["unit_id"], **c} for e in request["engines"] for c in e["cycles"]]
    return pd.DataFrame(rows)[RAW_COLUMNS]


def payload_from_frame(dataset: str, df: pd.DataFrame) -> dict:
    engines = []
    for unit, g in df.groupby("unit_id", sort=False):
        engines.append({"unit_id": int(unit), "cycles": [
            {c: (int(r[c]) if c == "cycle" else float(r[c])) for c in RAW_COLUMNS[1:]} for r in g[RAW_COLUMNS].to_dict("records")]})
    return {"dataset": dataset, "engines": engines}


def frame_to_raw_text(df: pd.DataFrame) -> bytes:
    lines = [" ".join([str(int(r[0])), str(int(r[1]))] + [repr(float(v)) for v in r[2:]]) for r in df[RAW_COLUMNS].to_numpy(object)]
    return ("\n".join(lines) + "\n").encode()


def assert_clean(response) -> None:
    text = response.text
    for token in FORBIDDEN_IN_RESPONSES:
        assert token not in text, f"response leaks {token!r}: {text[:300]}"


@pytest.fixture
def tampered_model_dir(tmp_path) -> Path:
    target = tmp_path / "final"
    shutil.copytree(MODEL_DIR, target)
    with open(target / "FD003" / "model.joblib", "ab") as fh:
        fh.write(b"\x00tampered")
    return target
