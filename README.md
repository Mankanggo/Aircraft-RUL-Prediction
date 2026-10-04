YO

## Local FastAPI inference API

The API (`src/api/`) wraps the frozen inference pipeline (`src/pipelines/prediction_pipeline.py`) without changing it. Requests are validated, converted to the raw 26-column format, and passed to the existing `RULPredictor`. That predictor applies the frozen preprocessing, the 290 engineered features and the frozen per-dataset model. The four models are loaded once at startup and verified before use:
- artifact SHA-256 hashes against `models/final/manifest.json`;
- installed library versions;
- consistency with the frozen `configs/final_config.json`.

### Run

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# interactive docs: http://127.0.0.1:8000/docs
```

### Settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `RUL_MODEL_DIR` | `<project>/models/final` | frozen model bundle |
| `RUL_CONFIG_DIR` | `<project>/configs` | frozen configuration |
| `RUL_MAX_ENGINES` / `RUL_MAX_TOTAL_ROWS` / `RUL_MAX_CYCLES_PER_ENGINE` | 500 / 100000 / 1000 | request limits (413 above) |
| `RUL_MAX_REQUEST_BYTES` / `RUL_MAX_UPLOAD_BYTES` | 64 MiB / 16 MiB | body / upload size limits |
| `RUL_MAX_CONCURRENT_PREDICTIONS` | 2 | simultaneous predictor calls; each frozen model uses 4 threads |
| `RUL_FAIL_FAST` | `true` | abort startup if verification fails; with `false`, `/ready` and predictions return 503 |
| `RUL_LOG_LEVEL` | `INFO` | JSON log lines on stdout |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: `{"status": "ok"}` |
| GET | `/ready` | 200 only when all four predictors loaded and passed hash/version/config checks, else 503 |
| GET | `/v1/models` | per-dataset model family, RUL cap, features, hashes, CV RMSE, supported operating conditions, required columns, limitations |
| POST | `/v1/predict` | RUL from raw JSON engine histories |
| POST | `/v1/predict/file` | RUL from a raw C-MAPSS text file (multipart: `dataset`, `file`) |

### Input contract (raw data only)

- **Required fields:** `dataset` (`FD001`–`FD004`), plus, for each engine, a `unit_id` (unique within the request) and its cycles.
- **Columns per cycle:** `cycle`, `op_setting_1..3` and all 21 sensors, named as in `src/data_schema.py`: `T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32`.
- **History:** each engine's history starts at cycle 1 and is contiguous. The prediction is made at the last supplied cycle.
- **No engineered features:** never send them; they are computed by the frozen pipeline.
- **Operating conditions:** FD001/FD003 models accept sea-level data only.

```bash
curl -s http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" -d '{
  "dataset": "FD001",
  "engines": [{"unit_id": 1, "cycles": [
    {"cycle": 1, "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
     "T2": 518.67, "T24": 641.82, "T30": 1589.70, "T50": 1400.60, "P2": 14.62, "P15": 21.61, "P30": 554.36,
     "Nf": 2388.06, "Nc": 9046.19, "epr": 1.30, "Ps30": 47.47, "phi": 521.66, "NRf": 2388.02, "NRc": 8138.62,
     "BPR": 8.4195, "farB": 0.03, "htBleed": 392, "Nf_dmd": 2388, "PCNfR_dmd": 100.0, "W31": 39.06, "W32": 23.4190}
  ]}]}'

curl -s http://127.0.0.1:8000/v1/predict/file -F dataset=FD001 -F file=@my_engines.txt
```

### Response and errors

- **Response:** `request_id`, `dataset`, model metadata (family, cap, config version and hash), and one prediction per engine, in request order. Each prediction has `unit_id`, `last_cycle`, the unrounded `predicted_rul`, an `at_or_near_cap` flag, a `short_history` flag and `warnings`.
- **Errors:** they return `{"request_id", "error": {"code", "message", "details?"}}`, with codes 422 (invalid request or data), 413 (over limits), 503 (not ready) and 500 (unexpected; logged server-side under the `request_id`).
- **Interpreting predictions:** a prediction at or near the cap means "at least about cap cycles". The `/v1/models` endpoint lists the documented limitations.

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/api        # API: lifecycle, validation/errors, golden fixtures, JSON/file parity vs RULPredictor, concurrency
pytest tests            # full suite
```

Regenerate the golden fixtures (truncated training engines, no labels) only if the frozen bundle changes: `python scripts/make_api_golden_fixture.py`.
