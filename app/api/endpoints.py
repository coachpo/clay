import logging
import uuid
from typing import Any, Dict, Optional

import litellm
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.conversion.request_converter import build_completion_request
from app.conversion.response_converter import (
    convert_openai_streaming_to_claude,
    convert_openai_to_claude_response,
    sse,
)
from app.core.config import config
from app.core.constants import Constants
from app.core.model_mapper import ModelMapper
from app.models.claude import ClaudeMessagesRequestModel, parse_claude_messages_request

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_ANTHROPIC_REQUEST_BYTES = 32 * 1024 * 1024
model_mapper = ModelMapper(config.big_model, config.middle_model, config.small_model)

litellm.drop_params = True


def _request_id_headers(request_id: str) -> Dict[str, str]:
    return {"request-id": request_id, "x-request-id": request_id}


def _anthropic_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _enforce_request_size_limit(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return

    try:
        content_length_value = int(content_length)
    except ValueError:
        return

    if content_length_value > MAX_ANTHROPIC_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=_anthropic_error_response(
                "Request body exceeds 32 MB limit", "request_too_large"
            ),
        )


async def _parse_claude_messages_request(http_request: Request) -> ClaudeMessagesRequestModel:
    try:
        payload = await http_request.json()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response(f"Invalid JSON body: {error}"),
        ) from error

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response("Invalid JSON body: expected an object."),
        )

    try:
        return parse_claude_messages_request(payload)
    except ValidationError as error:
        message = error.errors()[0]["msg"] if error.errors() else "Invalid request body"
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response(message),
        ) from error


async def validate_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
    content_type: Optional[str] = Header(None),
) -> None:
    _enforce_request_size_limit(request)

    if not anthropic_version:
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response("Missing required header: anthropic-version"),
        )

    if content_type and not content_type.startswith("application/json"):
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response("Content-Type must be application/json"),
        )

    if not config.validate_client_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail=_anthropic_error_response("Invalid API key", "authentication_error"),
        )


def _map_litellm_exception(error: Exception, request_id: str) -> HTTPException:
    if isinstance(error, litellm.AuthenticationError):
        status_code = 401
        error_type = "authentication_error"
    elif isinstance(error, litellm.RateLimitError):
        status_code = 429
        error_type = "rate_limit_error"
    elif isinstance(error, litellm.ContextWindowExceededError):
        status_code = 400
        error_type = "invalid_request_error"
    elif isinstance(error, litellm.Timeout):
        status_code = 408
        error_type = "timeout_error"
    elif isinstance(error, litellm.BadRequestError):
        status_code = 400
        error_type = "invalid_request_error"
    else:
        status_code = 500
        error_type = "api_error"

    return HTTPException(
        status_code=status_code,
        detail=_anthropic_error_response(str(error), error_type, request_id),
    )


@router.post("/v1/messages")
async def create_message(
    request: Request,
    _: None = Depends(validate_api_key),
) -> Any:
    request_id = str(uuid.uuid4())

    try:
        api_key = config.ensure_openai_api_key()
        claude_request = await _parse_claude_messages_request(request)
        completion_request = build_completion_request(
            claude_request,
            model_mapper,
            api_key=api_key,
            max_tokens_limit=config.max_tokens_limit,
            request_id=request_id,
            timeout=config.request_timeout,
        )

        if claude_request.stream:

            async def event_generator() -> Any:
                try:
                    stream = await litellm.acompletion(**completion_request)
                    async for event in convert_openai_streaming_to_claude(
                        stream,
                        claude_request,
                        request=request,
                    ):
                        yield event
                except Exception as error:
                    logger.exception("Streaming request failed")
                    http_error = _map_litellm_exception(error, request_id)
                    payload = http_error.detail
                    if isinstance(payload, dict):
                        yield sse(Constants.EVENT_ERROR, payload)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **_request_id_headers(request_id),
                },
            )

        response = await litellm.acompletion(**completion_request)
        response_dict = response.model_dump() if hasattr(response, "model_dump") else response
        claude_response = convert_openai_to_claude_response(response_dict, claude_request)
        return JSONResponse(content=claude_response, headers=_request_id_headers(request_id))
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=_anthropic_error_response(str(error), "invalid_request_error", request_id),
        ) from error
    except Exception as error:
        raise _map_litellm_exception(error, request_id) from error


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/")
async def root() -> Dict[str, str]:
    return {
        "service": "Clay API Proxy (LiteLLM)",
        "version": "2.0.0",
        "status": "running",
    }
