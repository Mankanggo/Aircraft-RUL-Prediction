"""
Request/response schemas. The per-cycle record is generated from src.data_schema, so the API
field names are exactly the raw columns the frozen inference pipeline expects (RAW_COLUMNS).
Clients send RAW sensor histories only - never engineered features.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.data_schema import SENSORS, SETTING_COLUMNS

Dataset = Literal["FD001", "FD002", "FD003", "FD004"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_STRICT = ConfigDict(strict=True, extra="forbid")

_cycle_fields: dict = {"cycle": (Annotated[int, Field(ge=1, description="Operational cycle number (1, 2, 3, ... per engine)")], ...)}
for _s in SETTING_COLUMNS:
    _cycle_fields[_s] = (Annotated[float, Field(allow_inf_nan=False, description="Operational setting (raw value)")], ...)
for _sensor in SENSORS:
    _unit = "" if _sensor.units == "--" else f" [{_sensor.units}]"
    _cycle_fields[_sensor.symbol] = (
        Annotated[float, Field(allow_inf_nan=False, description=f"sensor {_sensor.index}: {_sensor.description}{_unit}")], ...)

CycleRecord = create_model("CycleRecord", __config__=_STRICT, **_cycle_fields)
CycleRecord.__doc__ = "One raw C-MAPSS row of an engine: cycle, 3 operational settings and all 21 sensors."


class EngineHistory(BaseModel):
    """Full raw history of one engine, starting at cycle 1 with no gaps."""

    model_config = _STRICT
    unit_id: int
    cycles: list[CycleRecord] = Field(min_length=1)  # type: ignore[valid-type]


class PredictRequest(BaseModel):
    model_config = _STRICT
    dataset: Dataset = Field(description="Which frozen model to use (C-MAPSS sub-dataset of the engine fleet).")
    engines: list[EngineHistory] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_unit_ids(self) -> "PredictRequest":
        seen, dupes = set(), set()
        for e in self.engines:
            (dupes if e.unit_id in seen else seen).add(e.unit_id)
        if dupes:
            raise ValueError(f"duplicate unit_id values: {sorted(dupes)[:10]} (each engine must appear once)")
        return self


class ModelInfo(BaseModel):
    dataset: Dataset
    family: str
    rul_cap: float
    n_features: int
    config_version: str
    config_sha256: str


class EnginePrediction(BaseModel):
    unit_id: int
    last_cycle: int
    predicted_rul: float = Field(description="Predicted remaining cycles after last_cycle (unrounded).")
    at_or_near_cap: bool
    short_history: bool
    warnings: list[str]


class PredictResponse(BaseModel):
    request_id: str
    dataset: Dataset
    model: ModelInfo
    predictions: list[EnginePrediction]


class ModelCard(BaseModel):
    dataset: Dataset
    family: str
    rul_cap: float
    n_features: int
    params: dict
    trained_at: str
    artifact_sha256: dict[str, str]
    supported_operating_conditions: list[dict[str, float]]
    cv_rmse: dict[str, float]
    limitations: list[str]


class InputColumn(BaseModel):
    name: str
    description: str
    units: Optional[str] = None


class ModelsResponse(BaseModel):
    config_version: str
    config_sha256: str
    required_input_columns: list[InputColumn]
    input_rules: list[str]
    models: dict[str, ModelCard]


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    config_version: str
    config_sha256: str
    checks: dict[str, str]
    models: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
