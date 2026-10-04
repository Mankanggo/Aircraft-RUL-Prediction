"""Startup, readiness, metadata and integrity failures."""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model_registry import ArtifactIntegrityError
from src.data_schema import RAW_COLUMNS, SUBSETS
from tests.api.conftest import CONFIG_DIR, MODEL_DIR, assert_clean, default_settings


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["X-Request-ID"]


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert body["status"] == "ready" and body["config_sha256"] == manifest["config_body_sha256"]
    assert set(body["checks"].values()) == {"ok"} and set(body["models"]) == set(SUBSETS)


def test_models_metadata(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    config = json.loads((CONFIG_DIR / "final_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in body["required_input_columns"]] == RAW_COLUMNS
    assert body["config_version"] == config["version"]
    for s in SUBSETS:
        card = body["models"][s]
        assert card["family"] == config["models"][s]["family"]
        assert card["rul_cap"] == config["models"][s]["rul_cap"]
        assert card["n_features"] == config["feature_set"]["n_features"] == 290
        assert card["artifact_sha256"] == manifest["subsets"][s]["artifact_sha256"]
        assert card["limitations"]
    assert len(body["models"]["FD001"]["supported_operating_conditions"]) == 1
    assert len(body["models"]["FD004"]["supported_operating_conditions"]) == 6


def test_request_id_echo_and_json_404(client):
    r = client.get("/no-such-endpoint", headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 404 and r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123" and r.json()["error"]["code"] == "http_error"


def test_tampered_artifact_fails_startup(tampered_model_dir):
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch for FD003"):
        with TestClient(create_app(default_settings(model_dir=tampered_model_dir))):
            pass


def test_tampered_artifact_not_ready_when_not_fail_fast(tampered_model_dir, golden):
    with TestClient(create_app(default_settings(model_dir=tampered_model_dir, fail_fast=False))) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        assert "hash mismatch for FD003" in r.json()["error"]["message"]
        assert_clean(r)
        assert c.post("/v1/predict", json=golden["FD001"]["request"]).status_code == 503
        assert c.get("/v1/models").status_code == 503


def test_library_version_mismatch_fails(tmp_path):
    target = tmp_path / "final"
    shutil.copytree(MODEL_DIR, target)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["libraries"]["xgboost"] = "0.0.1"
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="library version mismatch: xgboost"):
        with TestClient(create_app(default_settings(model_dir=target))):
            pass


def test_config_mismatch_fails(tmp_path):
    cfg_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, cfg_dir)
    cfg = json.loads((cfg_dir / "final_config.json").read_text(encoding="utf-8"))
    cfg["models"]["FD001"]["rul_cap"] = 999.0  # edited after freezing
    (cfg_dir / "final_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="differs from its frozen hash"):
        with TestClient(create_app(default_settings(config_dir=cfg_dir))):
            pass


def test_missing_model_dir_fails_cleanly(tmp_path):
    with TestClient(create_app(default_settings(model_dir=tmp_path / "nowhere", fail_fast=False))) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "required file missing: model manifest" in r.json()["error"]["message"]
        assert_clean(r)


def test_config_check_is_line_ending_independent(tmp_path):
    """Configs committed with LF endings (git autocrlf) must still verify: content, not bytes, is checked."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name in ("final_config.json", "final_feature_list.json"):
        data = (CONFIG_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        (cfg_dir / name).write_bytes(data)
    with TestClient(create_app(default_settings(config_dir=cfg_dir))) as c:
        assert c.get("/ready").status_code == 200
