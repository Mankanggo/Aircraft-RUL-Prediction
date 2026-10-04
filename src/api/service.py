"""
Prediction service: API request -> 26-column raw DataFrame -> existing RULPredictor -> response.

No feature engineering or model logic lives here. Data validation is delegated to the frozen
validate_input(); this module only translates its outcome into API errors, enforces size limits,
checks that the chosen model supports the input's operating conditions, and maps the predictor's
results (sorted by unit_id) back to the requested engine order.
"""

import io
import threading
from typing import Optional

import pandas as pd

from src.api.errors import PayloadTooLargeError, RequestDataError, sanitize_message
from src.api.model_registry import LoadedModel, ModelRegistry, condition_settings
from src.api.schemas import EnginePrediction, ModelInfo, PredictRequest, PredictResponse
from src.api.settings import Settings
from src.data_schema import RAW_COLUMNS, SETTING_COLUMNS
from src.pipelines.prediction_pipeline import InputValidationError, validate_input
from src.utils import assign_operating_condition, read_raw_file

_RECORD_COLUMNS = RAW_COLUMNS[1:]  # everything except unit_id, in pipeline order


class PredictionService:
    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings
        # bounds simultaneous predictor calls (each frozen model uses 4 threads)
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_predictions)

    # -- inputs -------------------------------------------------------------------------
    def frame_from_request(self, request: PredictRequest) -> pd.DataFrame:
        """Schema-validated request -> raw DataFrame with exactly RAW_COLUMNS."""
        rows = [(engine.unit_id, *(getattr(record, col) for col in _RECORD_COLUMNS))
                for engine in request.engines for record in engine.cycles]
        return pd.DataFrame(rows, columns=RAW_COLUMNS)

    @staticmethod
    def frame_from_file(content: bytes) -> pd.DataFrame:
        """Raw C-MAPSS text (26 whitespace-separated columns) via the existing read_raw_file."""
        if not content.strip():
            raise RequestDataError("uploaded file is empty")
        try:
            return read_raw_file(io.BytesIO(content))
        except (ValueError, UnicodeDecodeError):  # wrong column count, ragged rows, undecodable bytes
            raise RequestDataError(f"malformed raw file: expected {len(RAW_COLUMNS)} whitespace-separated numeric "
                                   "columns per line (unit_id, cycle, 3 settings, 21 sensors)") from None

    def check_limits(self, n_engines: int, n_rows: int, max_cycles: int) -> None:
        s = self.settings
        if n_engines > s.max_engines:
            raise PayloadTooLargeError(f"too many engines: {n_engines} > {s.max_engines}")
        if n_rows > s.max_total_rows:
            raise PayloadTooLargeError(f"too many rows: {n_rows} > {s.max_total_rows}")
        if max_cycles > s.max_cycles_per_engine:
            raise PayloadTooLargeError(f"engine history too long: {max_cycles} > {s.max_cycles_per_engine} cycles")

    # -- prediction ---------------------------------------------------------------------
    def predict(self, dataset: str, df: pd.DataFrame, request_id: str, order: Optional[list] = None) -> PredictResponse:
        model: LoadedModel = self.registry.models[dataset]
        self._validate(df, model)
        with self._slots:
            result = model.predictor.predict_engines(df)  # frozen pipeline, unchanged
        by_unit = result.set_index("unit_id")
        order = list(order) if order is not None else list(result["unit_id"])
        predictions = [self._engine_prediction(int(unit), by_unit.loc[int(unit)], model) for unit in order]
        return PredictResponse(
            request_id=request_id, dataset=dataset,
            model=ModelInfo(dataset=dataset, family=model.family, rul_cap=model.rul_cap, n_features=model.n_features,
                            config_version=self.registry.config_version, config_sha256=self.registry.config_sha256),
            predictions=predictions,
        )

    def _validate(self, df: pd.DataFrame, model: LoadedModel) -> None:
        try:
            validate_input(df)
        except InputValidationError as exc:
            raise RequestDataError(sanitize_message(str(exc))) from None
        except ValueError:
            # validate_input's only other failure is assign_operating_condition: confirm, else re-raise (500)
            try:
                assign_operating_condition(df[SETTING_COLUMNS])
            except ValueError:
                raise RequestDataError("operational settings do not match any of the six known C-MAPSS operating "
                                       "conditions (see GET /v1/models)") from None
            raise
        present = set(int(c) for c in assign_operating_condition(df[SETTING_COLUMNS]).unique())
        unsupported = sorted(present - set(model.supported_conditions))
        if unsupported:
            supported = [condition_settings(c) for c in model.supported_conditions]
            raise RequestDataError(
                f"the {model.dataset} model was trained on operating condition(s) {supported} only; the input contains "
                f"{len(unsupported)} other condition(s). Multi-condition data requires FD002 or FD004.")

    def _engine_prediction(self, unit_id: int, row: pd.Series, model: LoadedModel) -> EnginePrediction:
        rul = float(row["predicted_rul"])
        last_cycle = int(row["last_cycle"])
        near_cap = rul >= model.rul_cap - self.settings.near_cap_margin
        short = last_cycle <= self.settings.short_history_cycles
        warnings = []
        if near_cap:
            warnings.append(f"prediction is at or near the model's RUL cap ({model.rul_cap:g}); "
                            "the true RUL may be substantially higher")
        if short:
            warnings.append(f"short history ({last_cycle} cycles); expect larger errors")
        return EnginePrediction(unit_id=unit_id, last_cycle=last_cycle, predicted_rul=rul,
                                at_or_near_cap=near_cap, short_history=short, warnings=warnings)
