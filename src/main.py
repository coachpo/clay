import sys
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.endpoints import router as api_router
from src.core.config import config

app = FastAPI(title="Clay API Proxy", version="1.0.0")
app.include_router(api_router)


def _request_id_headers(request_id: str) -> dict[str, str]:
    return {"request-id": request_id, "x-request-id": request_id}


def _is_anthropic_path(path: str) -> bool:
    return path.startswith("/v1/messages")


def _is_openai_path(path: str) -> bool:
    return path.startswith("/v1/models")


def _extract_request_id(detail: object) -> str:
    if isinstance(detail, dict):
        request_id = detail.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    return str(uuid.uuid4())


@app.exception_handler(HTTPException)
async def anthropic_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _extract_request_id(exc.detail)
    path = request.url.path

    if _is_openai_path(path):
        openai_payload: dict[str, Any]
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            openai_payload = exc.detail
        else:
            openai_payload = {
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                    "param": None,
                    "code": None,
                }
            }
        return JSONResponse(
            status_code=exc.status_code,
            content=openai_payload,
            headers=_request_id_headers(request_id),
        )

    if _is_anthropic_path(path):
        anthropic_payload: dict[str, Any]
        if isinstance(exc.detail, dict) and exc.detail.get("type") == "error":
            anthropic_payload = dict(exc.detail)
            anthropic_payload.setdefault("request_id", request_id)
        else:
            anthropic_payload = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error" if exc.status_code < 500 else "api_error",
                    "message": str(exc.detail),
                },
                "request_id": request_id,
            }
        return JSONResponse(
            status_code=exc.status_code,
            content=anthropic_payload,
            headers=_request_id_headers(request_id),
        )

    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, _: RequestValidationError) -> JSONResponse:
    request_id = str(uuid.uuid4())
    message = "Invalid request body"
    path = request.url.path

    if _is_openai_path(path):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
            headers=_request_id_headers(request_id),
        )

    if _is_anthropic_path(path):
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": message,
                },
                "request_id": request_id,
            },
            headers=_request_id_headers(request_id),
        )

    return JSONResponse(status_code=400, content={"detail": message})


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Clay API Proxy v1.0.0")
        print("")
        print("Usage: clay")
        print("")
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your OpenAI API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
        print("                      If set, clients must provide this exact API key")
        print("  OPENAI_BASE_URL - OpenAI API base URL " "(default: https://api.openai.com/v1)")
        print("  BIG_MODEL - Model for opus requests (default: gpt-4o)")
        print("  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)")
        print("  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
        print("  HOST - Server host (default: 0.0.0.0)")
        print("  PORT - Server port (default: 8000)")
        print("  LOG_LEVEL - Logging level (default: INFO)")
        print("  UVICORN_WORKERS - Uvicorn worker process count (default: 1)")
        print("  MAX_TOKENS_LIMIT - Token limit (default: 4096)")
        print("  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)")
        print("  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet/opus models -> {config.big_model}")
        sys.exit(0)

    print("Clay API Proxy v1.0.0")
    print("Configuration loaded successfully")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Uvicorn Workers: {config.uvicorn_workers}")
    print(f"   Server: {config.host}:{config.port}")
    print(
        "   Client API Key Validation: " f"{'Enabled' if config.anthropic_api_key else 'Disabled'}"
    )
    print("")

    log_level = config.log_level.split()[0].lower()
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        workers=config.uvicorn_workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
