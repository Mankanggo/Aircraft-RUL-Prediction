"""
FastAPI app for the frozen C-MAPSS RUL models.

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Request -> Pydantic validation -> PredictionService -> existing RULPredictor (frozen preprocessing,
features and model) -> validated response. The four predictors are loaded and verified once, at
startup; nothing is loaded or fitted per request.
"""

import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.errors import (
    INTERNAL_ERROR_MESSAGE, NotReadyError, PayloadTooLargeError, error_body, register_exception_handlers,
)
from src.api.logging_config import configure_logging
from src.api.model_registry import COMMON_LIMITATIONS, DATASET_NOTES, ArtifactIntegrityError, condition_settings, load_registry
from src.api.schemas import (
    Dataset, HealthResponse, InputColumn, ModelCard, ModelsResponse, PredictRequest, PredictResponse, ReadyResponse,
)
from src.api.service import PredictionService
from src.api.settings import Settings
from src.data_schema import RAW_COLUMNS, SENSOR_BY_SYMBOL

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

INPUT_RULES = [
    "Send RAW sensor rows only; the 290 engineered features are computed by the frozen pipeline.",
    "Each engine's history must start at cycle 1 and be contiguous (1, 2, 3, ...), without duplicate cycles.",
    "All 26 columns are required, including the 4 sensors the model does not use (T2, P2, Nf_dmd, PCNfR_dmd).",
    "Operational settings must match one of the six C-MAPSS operating conditions; FD001/FD003 accept sea level only.",
    "The prediction is made at each engine's last supplied cycle.",
]


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logger = configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = None
        app.state.service = None
        app.state.startup_error = None
        try:
            registry = load_registry(settings.model_dir, settings.config_dir)
        except Exception as exc:
            public = str(exc) if isinstance(exc, ArtifactIntegrityError) else "model loading failed"
            app.state.startup_error = public
            logger.exception("frozen model verification failed", extra={"detail": public})
            if settings.fail_fast:
                raise
        else:
            app.state.registry = registry
            app.state.service = PredictionService(registry, settings)
            logger.info("models loaded and verified", extra={
                "config_version": registry.config_version,
                "detail": {s: {"family": m.family, "rul_cap": m.rul_cap, "model_sha256": m.artifact_sha256["model.joblib"][:16]}
                           for s, m in registry.models.items()}})
        yield

    app = FastAPI(title="Aircraft Engine RUL API", version="1.0.0", lifespan=lifespan,
                  description="Remaining-useful-life predictions from raw NASA C-MAPSS engine histories, served by the "
                              "frozen classical models (classical-v1.0).")
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.log_extra = {}
        start = time.perf_counter()
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            request.state.error_code = "payload_too_large"
            response = JSONResponse(status_code=413, content=error_body(
                request_id, "payload_too_large", f"request body exceeds {settings.max_request_bytes} bytes"))
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("unhandled error", extra={"request_id": request_id, "endpoint": request.url.path})
                request.state.error_code = "internal_error"
                response = JSONResponse(status_code=500, content=error_body(request_id, "internal_error", INTERNAL_ERROR_MESSAGE))
        response.headers["X-Request-ID"] = request_id
        logger.info("request", extra={
            "request_id": request_id, "method": request.method, "endpoint": request.url.path, "status": response.status_code,
            "latency_ms": round(1000 * (time.perf_counter() - start), 1),
            "error_code": getattr(request.state, "error_code", None), **request.state.log_extra})
        return response

    def ready_service(request: Request) -> PredictionService:
        service = request.app.state.service
        if service is None:
            raise NotReadyError(f"models are not available: {request.app.state.startup_error or 'not loaded'}")
        return service

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    async def health() -> HealthResponse:
        """Liveness: the process is up. Does not touch the models."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["status"])
    async def ready(request: Request) -> ReadyResponse:
        """Readiness: all four frozen predictors loaded and passed hash/version/config checks."""
        registry = ready_service(request).registry
        return ReadyResponse(status="ready", config_version=registry.config_version, config_sha256=registry.config_sha256,
                             checks=registry.checks,
                             models={s: f"{m.family} (cap {m.rul_cap:g})" for s, m in registry.models.items()})

    @app.get("/v1/models", response_model=ModelsResponse, tags=["models"])
    async def models(request: Request) -> ModelsResponse:
        """Metadata of the four frozen models and the raw input contract."""
        registry = ready_service(request).registry
        columns = []
        for name in RAW_COLUMNS:
            sensor = SENSOR_BY_SYMBOL.get(name)
            if sensor:
                columns.append(InputColumn(name=name, description=f"sensor {sensor.index}: {sensor.description}",
                                           units=None if sensor.units == "--" else sensor.units))
            else:
                columns.append(InputColumn(name=name, description={"unit_id": "engine identifier (unique per request)",
                                                                   "cycle": "operational cycle, 1..n"}.get(name, "operational setting")))
        cards = {s: ModelCard(dataset=s, family=m.family, rul_cap=m.rul_cap, n_features=m.n_features, params=m.params,
                              trained_at=m.trained_at, artifact_sha256=m.artifact_sha256,
                              supported_operating_conditions=[condition_settings(c) for c in m.supported_conditions],
                              cv_rmse=m.cv_rmse, limitations=[DATASET_NOTES[s]] + COMMON_LIMITATIONS)
                 for s, m in registry.models.items()}
        return ModelsResponse(config_version=registry.config_version, config_sha256=registry.config_sha256,
                              required_input_columns=columns, input_rules=INPUT_RULES, models=cards)

    @app.post("/v1/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        """RUL for each engine at its last supplied cycle, from raw JSON histories."""
        service = ready_service(request)
        n_rows = sum(len(e.cycles) for e in payload.engines)
        request.state.log_extra = {"dataset": payload.dataset, "n_engines": len(payload.engines), "n_rows": n_rows}
        service.check_limits(len(payload.engines), n_rows, max(len(e.cycles) for e in payload.engines))
        df = service.frame_from_request(payload)
        response = service.predict(payload.dataset, df, request.state.request_id, order=[e.unit_id for e in payload.engines])
        request.state.log_extra |= {"model_family": response.model.family, "config_version": response.model.config_version}
        return response

    @app.post("/v1/predict/file", response_model=PredictResponse, tags=["prediction"])
    def predict_file(request: Request, dataset: Dataset = Form(...), file: UploadFile = File(...)) -> PredictResponse:
        """RUL for each engine in a raw C-MAPSS text file (26 whitespace-separated columns per row)."""
        service = ready_service(request)
        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise PayloadTooLargeError(f"uploaded file exceeds {settings.max_upload_bytes} bytes")
        df = service.frame_from_file(content)
        sizes = df.groupby("unit_id").size()
        request.state.log_extra = {"dataset": dataset, "n_engines": int(len(sizes)), "n_rows": int(len(df))}
        service.check_limits(len(sizes), len(df), int(sizes.max()) if len(sizes) else 0)
        order = [int(u) for u in df["unit_id"].dropna().drop_duplicates()]
        response = service.predict(dataset, df, request.state.request_id, order=order)
        request.state.log_extra |= {"model_family": response.model.family, "config_version": response.model.config_version}
        return response

    return app


app = create_app()
