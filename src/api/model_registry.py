"""
Load and verify the four frozen predictors once, at startup.

Loading itself is delegated to RULPredictor.load(..., verify=True), which checks every artifact
against the SHA-256 hashes in manifest.json. On top of that this module checks:
  * installed library versions == the versions the artifacts were trained with (manifest);
  * configs/final_config.json is the frozen config the manifest refers to (body hash), and the
    loaded models/features match it. These checks compare parsed content, so they do not depend
    on the files' line endings.
Any failure raises ArtifactIntegrityError with a message safe to show to clients (no paths).
"""

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.data_schema import OPERATING_CONDITIONS, SETTING_COLUMNS, SUBSETS
from src.pipelines.prediction_pipeline import RULPredictor

# manifest library key -> installed distribution name
_DISTRIBUTIONS = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "xgboost": "xgboost",
                  "lightgbm": "lightgbm", "joblib": "joblib"}

# Documented in reports/classical_ml_stage_report.md and reports/final_evaluation/FINAL_EVALUATION_REPORT.md
COMMON_LIMITATIONS = [
    "Predictions saturate at the training RUL cap: a prediction at or near the cap means 'at least about cap cycles', "
    "and true RULs above the cap are under-predicted (the main error source on FD002/FD004).",
    "Fast-degrading (short-lived) engines tend to be predicted late (roughly +5 to +10 cycles on average).",
    "Mid-range true RUL (about 51-100 cycles) shows a late bias of about +5 to +9 cycles.",
    "Short histories (few observed cycles) carry little degradation evidence and have larger errors.",
]
DATASET_NOTES = {
    "FD001": "Single operating condition (sea level), HPC degradation fault mode.",
    "FD002": "Six operating conditions, HPC degradation fault mode.",
    "FD003": "Single operating condition (sea level), HPC + fan degradation fault modes.",
    "FD004": "Six operating conditions, HPC + fan degradation fault modes.",
}


class ArtifactIntegrityError(RuntimeError):
    """Frozen artifacts missing, modified or inconsistent; message is safe for clients."""


@dataclass(frozen=True)
class LoadedModel:
    dataset: str
    predictor: RULPredictor
    family: str
    rul_cap: float
    n_features: int
    params: dict
    trained_at: str
    artifact_sha256: dict
    supported_conditions: tuple
    cv_rmse: dict


@dataclass(frozen=True)
class ModelRegistry:
    models: dict
    config_version: str
    config_sha256: str
    checks: dict


def _load_json(path: Path, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ArtifactIntegrityError(f"required file missing: {what}") from None
    except (OSError, json.JSONDecodeError):
        raise ArtifactIntegrityError(f"unreadable file: {what}") from None


def _check_libraries(manifest: dict) -> None:
    trained_py = manifest.get("python", "")
    if trained_py.split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ArtifactIntegrityError(f"Python {platform.python_version()} differs from the training version {trained_py}")
    for key, expected in manifest["libraries"].items():
        try:
            installed = version(_DISTRIBUTIONS.get(key, key))
        except PackageNotFoundError:
            raise ArtifactIntegrityError(f"required library not installed: {key}=={expected}") from None
        if installed != expected:
            raise ArtifactIntegrityError(f"library version mismatch: {key} {installed} installed, artifacts require {expected}")


def _check_config(config: dict, manifest: dict) -> None:
    body = {k: v for k, v in config.items() if k != "frozen"}
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    frozen_hash = config.get("frozen", {}).get("config_body_sha256")
    if body_hash != frozen_hash:
        raise ArtifactIntegrityError("final_config.json content differs from its frozen hash")
    if manifest.get("config_body_sha256") != frozen_hash or manifest.get("config_version") != config.get("version"):
        raise ArtifactIntegrityError("model manifest does not belong to the frozen configuration")


def load_registry(model_dir: Path, config_dir: Path) -> ModelRegistry:
    model_dir, config_dir = Path(model_dir), Path(config_dir)
    manifest = _load_json(model_dir / "manifest.json", "model manifest")
    config = _load_json(config_dir / "final_config.json", "final_config.json")
    feature_list = _load_json(config_dir / "final_feature_list.json", "final_feature_list.json")
    frozen_features = [f["feature"] for f in feature_list["features"]]

    _check_libraries(manifest)
    _check_config(config, manifest)

    models = {}
    for subset in SUBSETS:
        try:
            predictor = RULPredictor.load(subset, model_dir=str(model_dir), verify=True)
        except FileNotFoundError:
            raise ArtifactIntegrityError(f"missing artifact for {subset}") from None
        except RuntimeError as exc:
            if "hash" in str(exc):
                raise ArtifactIntegrityError(f"artifact hash mismatch for {subset}") from None
            raise
        meta, frozen = predictor.metadata, config["models"][subset]
        if (meta["family"], float(meta["rul_cap"]), meta["params"]) != (frozen["family"], float(frozen["rul_cap"]), frozen["params"]):
            raise ArtifactIntegrityError(f"{subset} model metadata differs from the frozen configuration")
        if list(predictor.preprocessor.feature_names_) != frozen_features or len(frozen_features) != config["feature_set"]["n_features"]:
            raise ArtifactIntegrityError(f"{subset} preprocessor features differ from the frozen feature list")
        models[subset] = LoadedModel(
            dataset=subset, predictor=predictor, family=meta["family"], rul_cap=float(meta["rul_cap"]),
            n_features=len(predictor.preprocessor.feature_names_), params=meta["params"],
            trained_at=manifest.get("trained_at", ""), artifact_sha256=manifest["subsets"][subset]["artifact_sha256"],
            supported_conditions=tuple(int(c) for c in predictor.preprocessor.normalizer_.stats_.index),
            cv_rmse={plan: float(v["rmse"]) for plan, v in config.get("cv_results", {}).get(subset, {}).items()},
        )
    checks = {"artifact_hashes": "ok", "library_versions": "ok", "config_consistency": "ok", "feature_list": "ok"}
    return ModelRegistry(models=models, config_version=config["version"], config_sha256=config["frozen"]["config_body_sha256"],
                         checks=checks)


def condition_settings(condition_id: int) -> dict:
    return dict(zip(SETTING_COLUMNS, OPERATING_CONDITIONS[condition_id]))
