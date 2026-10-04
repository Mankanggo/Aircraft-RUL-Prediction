"""
CRITICAL: the API must return exactly what the existing direct RULPredictor returns.

  * golden fixtures (committed; truncated training engines, no labels): API == stored expected
    == direct predictor, FD001-FD004
  * larger randomised parity on training data (if data/raw is available): JSON and file endpoints
  * concurrent requests against a real uvicorn server == sequential direct predictions
"""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
import pytest
import uvicorn

from src.api.main import create_app
from src.data_schema import RAW_COLUMNS, SUBSETS
from tests.api.conftest import default_settings, frame_to_raw_text, payload_from_frame, request_frame

TOL = 1e-6


@pytest.mark.parametrize("subset", SUBSETS)
def test_golden_fixture_parity(client, predictors, golden, subset):
    fixture = golden[subset]
    expected = pd.DataFrame(fixture["expected"]).set_index("unit_id")
    direct = predictors[subset].predict_engines(request_frame(fixture["request"])).set_index("unit_id")
    api = pd.DataFrame(client.post("/v1/predict", json=fixture["request"]).json()["predictions"]).set_index("unit_id")
    # frozen pipeline still produces the stored values ...
    np.testing.assert_allclose(direct.loc[expected.index, "predicted_rul"], expected.predicted_rul, atol=TOL, rtol=0)
    # ... and the API produces the direct values
    np.testing.assert_allclose(api.loc[direct.index, "predicted_rul"], direct.predicted_rul, atol=TOL, rtol=0)
    assert (api.loc[direct.index, "last_cycle"] == direct.last_cycle).all()


@pytest.mark.parametrize("subset", SUBSETS)
def test_golden_fixture_file_parity(client, predictors, golden, subset, tmp_path):
    df = request_frame(golden[subset]["request"])
    raw = frame_to_raw_text(df)
    path = tmp_path / "engines.txt"
    path.write_bytes(raw)
    direct = predictors[subset].predict_file(str(path)).set_index("unit_id")
    r = client.post("/v1/predict/file", data={"dataset": subset}, files={"file": ("engines.txt", raw)})
    assert r.status_code == 200, r.text
    api = pd.DataFrame(r.json()["predictions"]).set_index("unit_id")
    np.testing.assert_allclose(api.loc[direct.index, "predicted_rul"], direct.predicted_rul, atol=TOL, rtol=0)


def _truncated_training_engines(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for unit in rng.choice(train.unit_id.unique(), size=n, replace=False):
        engine = train[train.unit_id == unit]
        parts.append(engine[engine.cycle <= rng.integers(1, engine.cycle.max() + 1)])
    return pd.concat(parts)[RAW_COLUMNS]


@pytest.mark.parametrize("subset", SUBSETS)
def test_randomised_json_parity(client, predictors, cmapss, subset):
    df = _truncated_training_engines(cmapss[subset]["train"], 25, seed=int(subset[2:]))  # deterministic per subset
    direct = predictors[subset].predict_engines(df).set_index("unit_id")
    r = client.post("/v1/predict", json=payload_from_frame(subset, df))
    assert r.status_code == 200, r.text
    api = pd.DataFrame(r.json()["predictions"]).set_index("unit_id")
    assert len(api) == len(direct) == 25
    np.testing.assert_allclose(api.loc[direct.index, "predicted_rul"], direct.predicted_rul, atol=TOL, rtol=0)
    assert (api.loc[direct.index, "last_cycle"] == direct.last_cycle).all()


@pytest.mark.parametrize("subset", SUBSETS)
def test_original_raw_training_file_parity(client, predictors, subset):
    """The original raw text lines of real training engines through the file endpoint vs predict_file."""
    from src.utils import DEFAULT_RAW_DIR
    from tests.api.conftest import PROJECT_ROOT

    path = PROJECT_ROOT / DEFAULT_RAW_DIR / f"train_{subset}.txt"
    if not path.exists():
        pytest.skip("raw data not available")
    lines = path.read_bytes().splitlines(keepends=True)
    subset_lines = b"".join(ln for ln in lines if ln.split()[0] in {b"1", b"2", b"3", b"4", b"5"})
    direct = predictors[subset].predict_file(str(path)).set_index("unit_id").loc[[1, 2, 3, 4, 5]]
    r = client.post("/v1/predict/file", data={"dataset": subset}, files={"file": (path.name, subset_lines)})
    assert r.status_code == 200, r.text
    api = pd.DataFrame(r.json()["predictions"]).set_index("unit_id")
    np.testing.assert_allclose(api.loc[direct.index, "predicted_rul"], direct.predicted_rul, atol=TOL, rtol=0)


@pytest.fixture(scope="module")
def live_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(default_settings()), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_concurrent_requests_consistent(live_server, predictors, golden):
    """32 concurrent HTTP requests over all four models return exactly the direct predictions."""
    expected = {s: predictors[s].predict_engines(request_frame(golden[s]["request"])).set_index("unit_id").predicted_rul
                for s in SUBSETS}
    with httpx.Client(base_url=live_server, timeout=60) as http:
        assert http.get("/ready").status_code == 200

        def call(subset):
            r = http.post("/v1/predict", json=golden[subset]["request"])
            assert r.status_code == 200, r.text
            return subset, pd.DataFrame(r.json()["predictions"]).set_index("unit_id").predicted_rul

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, SUBSETS * 8))
    for subset, got in results:
        np.testing.assert_allclose(got.loc[expected[subset].index], expected[subset], atol=TOL, rtol=0)
