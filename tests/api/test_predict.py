"""/v1/predict and /v1/predict/file: valid requests, ordering, warnings, and every error path."""

import copy

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from tests.api.conftest import assert_clean, default_settings, frame_to_raw_text, request_frame


def _post(client, payload):
    r = client.post("/v1/predict", json=payload)
    assert_clean(r)
    return r


def test_valid_prediction_shape(client, golden):
    req = golden["FD004"]["request"]
    r = _post(client, req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "FD004" and body["request_id"] == r.headers["X-Request-ID"]
    assert body["model"] == {"dataset": "FD004", "family": "lightgbm", "rul_cap": 140.0, "n_features": 290,
                             "config_version": "classical-v1.0", "config_sha256": body["model"]["config_sha256"]}
    assert [p["unit_id"] for p in body["predictions"]] == [e["unit_id"] for e in req["engines"]]  # request order
    for p, e in zip(body["predictions"], req["engines"]):
        assert p["last_cycle"] == len(e["cycles"]) and p["predicted_rul"] >= 0
        assert p["short_history"] == (p["last_cycle"] <= 30)
        assert p["at_or_near_cap"] == (p["predicted_rul"] >= 140 - 5)
        assert bool(p["warnings"]) == (p["short_history"] or p["at_or_near_cap"])


def test_response_order_follows_request_not_sorted(client, golden):
    req = copy.deepcopy(golden["FD001"]["request"])
    req["engines"] = sorted(req["engines"], key=lambda e: -e["unit_id"])
    r = _post(client, req)
    assert [p["unit_id"] for p in r.json()["predictions"]] == [e["unit_id"] for e in req["engines"]]


@pytest.mark.parametrize("mutate, status, code", [
    (lambda r: r.pop("engines"), 422, "validation_error"),                                   # invalid request
    (lambda r: r.update(engines=[]), 422, "validation_error"),                               # no engines
    (lambda r: r["engines"][0]["cycles"][0].pop("T50"), 422, "validation_error"),            # missing sensor
    (lambda r: r["engines"][0]["cycles"][0].update(T50="1500"), 422, "validation_error"),   # string number
    (lambda r: r["engines"][0]["cycles"][0].update(Ps30=True), 422, "validation_error"),    # boolean
    (lambda r: r["engines"][0]["cycles"][0].update(cycle=1.5), 422, "validation_error"),    # non-integer cycle
    (lambda r: r["engines"][0].update(unit_id="7"), 422, "validation_error"),               # string unit id
    (lambda r: r["engines"][0]["cycles"][0].update(extra_feature=1.0), 422, "validation_error"),  # engineered features not accepted
    (lambda r: r.update(dataset="FD005"), 422, "validation_error"),                          # unsupported dataset
    (lambda r: r["engines"].append(copy.deepcopy(r["engines"][0])), 422, "validation_error"),  # duplicate unit ids
])
def test_schema_errors(client, golden, mutate, status, code):
    req = copy.deepcopy(golden["FD001"]["request"])
    mutate(req)
    r = _post(client, req)
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    for detail in r.json()["error"].get("details", []):  # submitted values are never echoed back
        assert set(detail) == {"loc", "msg", "type"}


def test_non_json_body(client):
    r = client.post("/v1/predict", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_nan_and_infinity_rejected(client):
    for token in (b"NaN", b"Infinity"):
        body = (b'{"dataset":"FD001","engines":[{"unit_id":1,"cycles":[{"cycle":1,"op_setting_1":0,"op_setting_2":0,'
                b'"op_setting_3":100,"T2":518.67,"T24":641.8,"T30":1589.7,"T50":' + token +
                b',"P2":14.62,"P15":21.61,"P30":554.4,"Nf":2388.0,"Nc":9046.2,"epr":1.3,"Ps30":47.47,"phi":521.7,'
                b'"NRf":2388.0,"NRc":8138.6,"BPR":8.42,"farB":0.03,"htBleed":392,"Nf_dmd":2388,"PCNfR_dmd":100.0,'
                b'"W31":39.06,"W32":23.42}]}]}')
        r = client.post("/v1/predict", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 422, (token, r.text)


def _engine_cycles(golden, subset="FD001", idx=1):
    return copy.deepcopy(golden[subset]["request"]["engines"][idx]["cycles"])


@pytest.mark.parametrize("edit, fragment", [
    (lambda cyc: cyc.append(copy.deepcopy(cyc[-1])), "duplicate (unit_id, cycle)"),           # duplicate cycle
    (lambda cyc: cyc.pop(5), "non-contiguous"),                                                   # gap
    (lambda cyc: cyc.pop(0), "not starting at cycle 1"),                                          # history not from 1
])
def test_pipeline_validation_errors(client, golden, edit, fragment):
    cycles = _engine_cycles(golden)
    edit(cycles)
    r = _post(client, {"dataset": "FD001", "engines": [{"unit_id": 5, "cycles": cycles}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    assert fragment in r.json()["error"]["message"]


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
def test_multi_condition_data_rejected_by_single_condition_models(client, golden, subset):
    req = copy.deepcopy(golden["FD004"]["request"])  # six-condition data
    req["dataset"] = subset
    r = _post(client, req)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    assert "FD002 or FD004" in r.json()["error"]["message"]


def test_unknown_operating_condition_rejected(client, golden):
    req = copy.deepcopy(golden["FD002"]["request"])
    req["engines"][0]["cycles"][3]["op_setting_1"] = 33.0
    r = _post(client, req)
    assert r.status_code == 422 and "six known" in r.json()["error"]["message"]


def test_oversized_requests():
    small = default_settings(max_engines=2, max_total_rows=150, max_cycles_per_engine=100, max_request_bytes=200_000,
                             max_upload_bytes=1_000)
    import json as _json
    from tests.api.conftest import GOLDEN_DIR
    req = _json.loads((GOLDEN_DIR / "FD001.json").read_text(encoding="utf-8"))["request"]
    with TestClient(create_app(small)) as c:
        r = c.post("/v1/predict", json=req)                                   # 4 engines > 2
        assert r.status_code == 413 and "too many engines" in r.json()["error"]["message"]
        r = c.post("/v1/predict", json={"dataset": "FD001", "engines": req["engines"][1:3]})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"  # rows or cycles limit
        big = {"dataset": "FD001", "engines": [dict(req["engines"][1], cycles=req["engines"][1]["cycles"] * 40)]}
        assert len(_json.dumps(big)) > 200_000
        r = c.post("/v1/predict", content=_json.dumps(big).encode(), headers={"content-type": "application/json"})
        assert r.status_code == 413 and "request body exceeds" in r.json()["error"]["message"]
        r = c.post("/v1/predict/file", data={"dataset": "FD001"}, files={"file": ("x.txt", b"1 1 " * 600)})
        assert r.status_code == 413 and "uploaded file exceeds" in r.json()["error"]["message"]


@pytest.mark.parametrize("content, fragment", [
    (b"", "empty"),
    (b"1 1 0.0 0.0 100.0 518.67\n", "malformed raw file"),                     # too few columns
    (b"1 1 " + b"0.5 " * 24 + b"\n1 2 " + b"0.5 " * 25 + b"\n", "malformed raw file"),  # ragged rows
    (b"\xff\xfe\x00garbage\x00\n", "malformed raw file"),                      # binary
])
def test_malformed_raw_file(client, content, fragment):
    r = client.post("/v1/predict/file", data={"dataset": "FD001"}, files={"file": ("bad.txt", content)})
    assert_clean(r)
    assert r.status_code == 422 and fragment in r.json()["error"]["message"]


def test_raw_file_with_non_numeric_token(client, golden):
    raw = frame_to_raw_text(request_frame(golden["FD001"]["request"])).replace(b"518.67", b"abc", 1)
    r = client.post("/v1/predict/file", data={"dataset": "FD001"}, files={"file": ("bad.txt", raw)})
    assert r.status_code == 422 and "non-numeric" in r.json()["error"]["message"]


def test_file_endpoint_requires_dataset(client, golden):
    raw = frame_to_raw_text(request_frame(golden["FD001"]["request"]))
    assert client.post("/v1/predict/file", files={"file": ("x.txt", raw)}).status_code == 422
    assert client.post("/v1/predict/file", data={"dataset": "FD009"}, files={"file": ("x.txt", raw)}).status_code == 422


def test_unexpected_error_is_500_without_internals(golden, monkeypatch):
    with TestClient(create_app(default_settings())) as c:
        service = c.app.state.service
        predictor = service.registry.models["FD001"].predictor

        def boom(df):
            raise RuntimeError("secret internal detail at C:\\somewhere\\file.py")

        monkeypatch.setattr(predictor, "predict_engines", boom)
        r = c.post("/v1/predict", json=golden["FD001"]["request"], headers={"X-Request-ID": "req-500"})
        assert r.status_code == 500
        assert r.json() == {"request_id": "req-500", "error": {"code": "internal_error",
                            "message": "An unexpected internal error occurred. Quote the request_id when reporting it."}}
        assert "secret" not in r.text
