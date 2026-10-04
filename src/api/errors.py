"""API error types and handlers. Responses never contain paths, stack traces or internals."""

import re
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class RequestDataError(APIError):
    """Well-formed request whose data violates the inference contract (422)."""

    status_code = 422
    code = "invalid_input"


class PayloadTooLargeError(APIError):
    status_code = 413
    code = "payload_too_large"


class NotReadyError(APIError):
    status_code = 503
    code = "not_ready"


INTERNAL_ERROR_MESSAGE = "An unexpected internal error occurred. Quote the request_id when reporting it."

_NUMPY_SCALAR = re.compile(r"np\.(?:int|float|uint|bool_?)\d*\(([^()]*)\)")


def sanitize_message(message: str) -> str:
    """Strip numpy scalar wrappers, e.g. 'np.int64(3)' -> '3', from frozen-pipeline messages."""
    return _NUMPY_SCALAR.sub(r"\1", message)


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_body(request_id: str, code: str, message: str, details: Optional[list] = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"request_id": request_id, "error": error}


def error_response(request: Request, status_code: int, code: str, message: str, details: Optional[list] = None) -> JSONResponse:
    request.state.error_code = code
    return JSONResponse(status_code=status_code, content=error_body(request_id_of(request), code, message, details))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # loc/msg/type only: never echo the submitted values back
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
                   for e in exc.errors()[:50]]
        return error_response(request, 422, "validation_error", "Request does not match the input schema.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return error_response(request, exc.status_code, "http_error", message)
