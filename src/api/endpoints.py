import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from src.conversion.request_converter import (
    convert_claude_to_openai,
    parse_tool_result_content,
)
from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.core.client import OpenAIClient
from src.core.config import config
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.models.claude import (
    ClaudeMessagesRequestModel,
    ClaudeTokenCountRequestModel,
    parse_claude_messages_request,
    parse_claude_token_count_request,
)
from src.models.openai import OpenAIChatCompletionsRequest, OpenAIResponsesRequest

router = APIRouter()

openai_api_key = config.openai_api_key or ""
custom_headers = config.get_custom_headers()
openai_client = OpenAIClient(
    openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    config.max_retries,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)


MAX_ANTHROPIC_REQUEST_BYTES = 32 * 1024 * 1024


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


def _openai_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _http_exception_from_message(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=_anthropic_error_response(message, error_type=error_type),
    )


def _extract_error_message(detail: Any) -> str:
    if isinstance(detail, dict):
        error_block = detail.get("error")
        if isinstance(error_block, dict):
            message = error_block.get("message")
            if isinstance(message, str) and message:
                return message
    return str(detail)


def _request_id_headers(request_id: str) -> Dict[str, str]:
    return {"request-id": request_id, "x-request-id": request_id}


def _extract_client_api_key(
    x_api_key: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    if x_api_key:
        return x_api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "", 1)
    return None


def _anthropic_error_type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        return "request_too_large"
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 529:
        return "overloaded_error"
    if status_code < 500:
        return "invalid_request_error"
    return "api_error"


def _openai_error_type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _is_supported_openai_model_id(model_id: str) -> bool:
    configured_model_ids = {
        config.big_model,
        config.middle_model,
        config.small_model,
    }
    if model_id in configured_model_ids:
        return True

    normalized_model_id = model_id.lower()
    return normalized_model_id.startswith(
        ("gpt-", "o1-", "o3-", "o4-", "gpt-5", "ep-", "doubao-", "deepseek-")
    )


def _enforce_request_size_limit(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return

    try:
        content_length_value = int(content_length)
    except ValueError:
        return

    if content_length_value > MAX_ANTHROPIC_REQUEST_BYTES:
        raise _http_exception_from_message(
            status_code=413,
            message="Request body exceeds 32 MB limit",
            error_type="request_too_large",
        )


def _resolve_requested_anthropic_version(
    anthropic_version: Optional[str],
    request_id: Optional[str] = None,
) -> str:
    supported = sorted(set(config.anthropic_supported_versions))
    default_version = config.anthropic_default_version

    if not anthropic_version:
        if config.anthropic_allow_missing_version:
            return default_version
        raise _http_exception_from_message(
            status_code=400,
            message="Missing required header: anthropic-version",
            error_type="invalid_request_error",
        )

    if anthropic_version in supported:
        return anthropic_version

    if config.anthropic_allow_version_fallback and supported:
        candidate = anthropic_version.strip()
        if _looks_like_version(candidate):
            fallback = _resolve_fallback_version(candidate, supported)
            if fallback:
                logger.warning(
                    "Unsupported anthropic-version '%s' requested; falling back to '%s'",
                    anthropic_version,
                    fallback,
                )
                return fallback

    expected_versions = ", ".join(supported) if supported else default_version
    raise _http_exception_from_message(
        status_code=400,
        message=(
            f"Unsupported anthropic-version '{anthropic_version}'. "
            f"Supported versions: {expected_versions}."
        ),
        error_type="invalid_request_error",
    )


def _looks_like_version(version: str) -> bool:
    parts = version.split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)


def _resolve_fallback_version(
    requested_version: str,
    supported_versions: List[str],
) -> Optional[str]:
    lower_or_equal = [version for version in supported_versions if version <= requested_version]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(supported_versions) if supported_versions else None


def _build_openai_request_from_claude(
    claude_request: ClaudeMessagesRequestModel,
    http_request: Request,
) -> Dict[str, Any]:
    openai_request = convert_claude_to_openai(claude_request, model_manager)

    extra_headers = openai_request.get("extra_headers")
    if not isinstance(extra_headers, dict):
        extra_headers = {}

    anthropic_beta = getattr(http_request.state, "anthropic_beta", None)
    if isinstance(anthropic_beta, str) and anthropic_beta:
        extra_headers["anthropic-beta"] = anthropic_beta

    resolved_anthropic_version = getattr(http_request.state, "resolved_anthropic_version", None)
    if isinstance(resolved_anthropic_version, str) and resolved_anthropic_version:
        extra_headers["anthropic-version"] = resolved_anthropic_version

    if extra_headers:
        openai_request["extra_headers"] = extra_headers

    _apply_anthropic_compat_enhancements(openai_request, claude_request)
    return openai_request


def _apply_anthropic_compat_enhancements(
    openai_request: Dict[str, Any],
    claude_request: ClaudeMessagesRequestModel,
) -> None:
    if not _should_attach_extra_body_metadata():
        return

    metadata_payload: Dict[str, Any] = {}
    if config.include_original_anthropic_request:
        metadata_payload["original_anthropic_request"] = claude_request.model_dump(
            exclude_none=True
        )

    extension_fields: Dict[str, Any] = {}
    if claude_request.top_k is not None:
        extension_fields["top_k"] = claude_request.top_k
    if claude_request.inference_geo is not None:
        extension_fields["inference_geo"] = claude_request.inference_geo
    if claude_request.thinking is not None:
        extension_fields["thinking"] = claude_request.thinking.model_dump(exclude_none=True)
    if extension_fields:
        metadata_payload["anthropic_extensions"] = extension_fields

    if not metadata_payload:
        return

    existing_extra_body = openai_request.get("extra_body")
    if isinstance(existing_extra_body, dict):
        target_extra_body = dict(existing_extra_body)
    else:
        target_extra_body = {}

    existing_proxy_meta = target_extra_body.get("proxy_metadata")
    if isinstance(existing_proxy_meta, dict):
        merged_proxy_meta = dict(existing_proxy_meta)
        merged_proxy_meta.update(metadata_payload)
    else:
        merged_proxy_meta = metadata_payload

    target_extra_body["proxy_metadata"] = merged_proxy_meta
    openai_request["extra_body"] = target_extra_body


def _should_attach_extra_body_metadata() -> bool:
    return config.allow_openai_extension_passthrough or config.include_original_anthropic_request


async def _parse_claude_messages_request(
    http_request: Request,
) -> ClaudeMessagesRequestModel:
    try:
        payload = await http_request.json()
    except Exception as error:
        raise _http_exception_from_message(
            status_code=400,
            message=f"Invalid JSON body: {error}",
            error_type="invalid_request_error",
        ) from error

    if not isinstance(payload, dict):
        raise _http_exception_from_message(
            status_code=400,
            message="Invalid JSON body: expected an object.",
            error_type="invalid_request_error",
        )

    try:
        return parse_claude_messages_request(
            payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
        )
    except ValidationError as error:
        raise HTTPException(status_code=400, detail={"errors": error.errors()}) from error


async def _parse_claude_token_count_request(
    http_request: Request,
) -> ClaudeTokenCountRequestModel:
    try:
        payload = await http_request.json()
    except Exception as error:
        raise _http_exception_from_message(
            status_code=400,
            message=f"Invalid JSON body: {error}",
            error_type="invalid_request_error",
        ) from error

    if not isinstance(payload, dict):
        raise _http_exception_from_message(
            status_code=400,
            message="Invalid JSON body: expected an object.",
            error_type="invalid_request_error",
        )

    try:
        return parse_claude_token_count_request(
            payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
        )
    except ValidationError as error:
        raise HTTPException(status_code=400, detail={"errors": error.errors()}) from error


async def validate_api_contract(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None),
    anthropic_beta: Optional[str] = Header(None),
    content_type: Optional[str] = Header(None, alias="content-type"),
) -> None:
    _enforce_request_size_limit(request)

    resolved_anthropic_version = _resolve_requested_anthropic_version(
        anthropic_version=anthropic_version,
    )
    request.state.resolved_anthropic_version = resolved_anthropic_version

    if content_type and not content_type.lower().startswith("application/json"):
        raise _http_exception_from_message(
            status_code=400,
            message="Unsupported Content-Type. Expected application/json.",
        )

    if anthropic_beta:
        request.state.anthropic_beta = anthropic_beta

    client_api_key = _extract_client_api_key(x_api_key, authorization)

    if not config.anthropic_api_key:
        return

    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning("Invalid API key provided by client")
        raise _http_exception_from_message(
            status_code=401,
            message="Invalid API key. Please provide a valid Anthropic API key.",
            error_type="authentication_error",
        )


async def validate_openai_api_contract(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> None:
    client_api_key = _extract_client_api_key(x_api_key, authorization)
    if not client_api_key:
        raise HTTPException(
            status_code=401,
            detail=_openai_error_response(
                "Missing API key. Provide Authorization: Bearer <key>.",
                error_type="authentication_error",
                code="invalid_api_key",
            ),
        )

    if config.anthropic_api_key and not config.validate_client_api_key(client_api_key):
        raise HTTPException(
            status_code=401,
            detail=_openai_error_response(
                "Invalid API key provided.",
                error_type="authentication_error",
                code="invalid_api_key",
            ),
        )


@router.post("/v1/messages")
async def create_message(
    http_request: Request,
    _: None = Depends(validate_api_contract),
) -> Any:
    request_id = str(uuid.uuid4())

    try:
        request = await _parse_claude_messages_request(http_request)
        logger.debug(
            "Processing Claude request: model=%s, stream=%s",
            request.model,
            request.stream,
        )

        openai_request = _build_openai_request_from_claude(request, http_request)

        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            openai_stream = openai_client.create_chat_completion_stream(openai_request, request_id)
            return StreamingResponse(
                convert_openai_streaming_to_claude_with_cancellation(
                    openai_stream,
                    request,
                    logger,
                    http_request,
                    openai_client,
                    request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    **_request_id_headers(request_id),
                },
            )

        openai_response = await _create_chat_completion_with_disconnect_cancellation(
            openai_request,
            http_request,
            request_id,
        )
        claude_response = convert_openai_to_claude_response(openai_response, request)
        return JSONResponse(content=claude_response, headers=_request_id_headers(request_id))

    except ValueError as error:
        payload = _anthropic_error_response(
            message=str(error),
            error_type="invalid_request_error",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=400,
            content=payload,
            headers=_request_id_headers(request_id),
        )
    except HTTPException as error:
        message = _extract_error_message(error.detail)
        if isinstance(error.detail, dict) and error.detail.get("type") == "error":
            payload = dict(error.detail)
            payload.setdefault("request_id", request_id)
        else:
            payload = _anthropic_error_response(
                message=message,
                error_type=_anthropic_error_type_for_status(error.status_code),
                request_id=request_id,
            )
        return JSONResponse(
            status_code=error.status_code,
            content=payload,
            headers=_request_id_headers(request_id),
        )
    except Exception as error:
        logger.exception("Unexpected error processing request: %s", error)
        payload = _anthropic_error_response(
            message=openai_client.classify_openai_error(str(error)),
            error_type="api_error",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=payload,
            headers=_request_id_headers(request_id),
        )


async def _create_chat_completion_with_disconnect_cancellation(
    openai_request: Dict[str, Any],
    http_request: Request,
    request_id: str,
) -> Dict[str, Any]:
    completion_task = asyncio.create_task(
        openai_client.create_chat_completion(openai_request, request_id)
    )
    disconnect_task = asyncio.create_task(_wait_for_http_disconnect(http_request))

    try:
        done, pending = await asyncio.wait(
            {completion_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for pending_task in pending:
            pending_task.cancel()

        if disconnect_task in done and disconnect_task.result():
            openai_client.cancel_request(request_id)
            completion_task.cancel()
            raise HTTPException(
                status_code=499,
                detail=_anthropic_error_response(
                    message="Client disconnected",
                    error_type="cancelled",
                    request_id=request_id,
                ),
            )

        return await completion_task
    finally:
        for task in (completion_task, disconnect_task):
            if not task.done():
                task.cancel()


async def _wait_for_http_disconnect(http_request: Request) -> bool:
    while True:
        if await http_request.is_disconnected():
            return True
        await asyncio.sleep(0.1)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    http_request: Request,
    _: None = Depends(validate_api_contract),
) -> Any:
    request_id = str(uuid.uuid4())

    try:
        request = await _parse_claude_token_count_request(http_request)
        estimated_tokens = _estimate_request_input_tokens(request)
        return JSONResponse(
            content={"input_tokens": estimated_tokens},
            headers=_request_id_headers(request_id),
        )
    except HTTPException as error:
        message = _extract_error_message(error.detail)
        payload = _anthropic_error_response(
            message=message,
            error_type=_anthropic_error_type_for_status(error.status_code),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=error.status_code,
            content=payload,
            headers=_request_id_headers(request_id),
        )
    except Exception as error:
        logger.exception("Error counting tokens: %s", error)
        payload = _anthropic_error_response(
            message=str(error),
            error_type="api_error",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=payload,
            headers=_request_id_headers(request_id),
        )


@router.post("/v1/chat/completions")
async def create_chat_completion_openai(
    request: OpenAIChatCompletionsRequest,
    _: None = Depends(validate_openai_api_contract),
) -> Any:
    request_id = str(uuid.uuid4())
    payload = request.model_dump(exclude_none=True)

    try:
        if request.stream:
            openai_stream = openai_client.create_chat_completion_stream(payload, request_id)
            return StreamingResponse(
                _format_openai_sse_stream(openai_stream),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **_request_id_headers(request_id),
                },
            )

        openai_response = await openai_client.create_chat_completion(payload, request_id)
        return JSONResponse(content=openai_response, headers=_request_id_headers(request_id))
    except HTTPException as error:
        error_detail = error.detail
        if isinstance(error_detail, dict) and isinstance(error_detail.get("error"), dict):
            upstream_error = error_detail["error"]
            payload = _openai_error_response(
                message=_extract_error_message(error_detail),
                error_type=_openai_error_type_for_status(error.status_code),
                param=upstream_error.get("param"),
                code=upstream_error.get("code"),
            )
        else:
            payload = _openai_error_response(
                message=_extract_error_message(error_detail),
                error_type=_openai_error_type_for_status(error.status_code),
            )
        return JSONResponse(
            status_code=error.status_code,
            content=payload,
            headers=_request_id_headers(request_id),
        )
    except Exception as error:
        payload = _openai_error_response(
            message=str(error),
            error_type="server_error",
        )
        return JSONResponse(
            status_code=500,
            content=payload,
            headers=_request_id_headers(request_id),
        )


@router.post("/v1/responses")
async def create_responses_openai(
    request: OpenAIResponsesRequest,
    _: None = Depends(validate_openai_api_contract),
) -> Any:
    request_id = str(uuid.uuid4())
    input_payload = request.input

    if input_payload is None:
        return JSONResponse(
            status_code=400,
            content=_openai_error_response("Field 'input' is required", param="input"),
            headers=_request_id_headers(request_id),
        )

    chat_messages = _responses_input_to_chat_messages(input_payload)
    if not chat_messages:
        return JSONResponse(
            status_code=400,
            content=_openai_error_response(
                "Field 'input' must contain at least one message",
                param="input",
            ),
            headers=_request_id_headers(request_id),
        )

    chat_request = _responses_request_to_chat_request(request, chat_messages)
    try:
        if request.stream:
            openai_stream = openai_client.create_chat_completion_stream(chat_request, request_id)
            return StreamingResponse(
                _openai_chat_stream_to_responses_events(openai_stream),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **_request_id_headers(request_id),
                },
            )

        openai_response = await openai_client.create_chat_completion(chat_request, request_id)
        return JSONResponse(
            content=_chat_completion_to_responses_payload(openai_response),
            headers=_request_id_headers(request_id),
        )
    except HTTPException as error:
        error_detail = error.detail
        if isinstance(error_detail, dict) and isinstance(error_detail.get("error"), dict):
            upstream_error = error_detail["error"]
            payload = _openai_error_response(
                message=_extract_error_message(error_detail),
                error_type=_openai_error_type_for_status(error.status_code),
                param=upstream_error.get("param"),
                code=upstream_error.get("code"),
            )
        else:
            payload = _openai_error_response(
                message=_extract_error_message(error_detail),
                error_type=_openai_error_type_for_status(error.status_code),
            )
        return JSONResponse(
            status_code=error.status_code,
            content=payload,
            headers=_request_id_headers(request_id),
        )
    except Exception as error:
        payload = _openai_error_response(
            message=str(error),
            error_type="server_error",
        )
        return JSONResponse(
            status_code=500,
            content=payload,
            headers=_request_id_headers(request_id),
        )


@router.get("/v1/models")
async def list_models_openai(_: None = Depends(validate_openai_api_contract)) -> Any:
    request_id = str(uuid.uuid4())
    created_at = int(datetime.now().timestamp())
    model_ids = [
        config.big_model,
        config.middle_model,
        config.small_model,
    ]
    unique_model_ids = list(dict.fromkeys(model_ids))
    payload = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created_at,
                "owned_by": "clay",
            }
            for model_id in unique_model_ids
        ],
    }
    return JSONResponse(content=payload, headers=_request_id_headers(request_id))


@router.get("/v1/models/{model_id}")
async def get_model_openai(model_id: str, _: None = Depends(validate_openai_api_contract)) -> Any:
    request_id = str(uuid.uuid4())
    if not _is_supported_openai_model_id(model_id):
        return JSONResponse(
            status_code=404,
            content=_openai_error_response(
                message=f"The model '{model_id}' does not exist",
                error_type="invalid_request_error",
                param="model",
                code="model_not_found",
            ),
            headers=_request_id_headers(request_id),
        )

    payload = {
        "id": model_id,
        "object": "model",
        "created": int(datetime.now().timestamp()),
        "owned_by": "clay",
    }
    return JSONResponse(content=payload, headers=_request_id_headers(request_id))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection() -> Any:
    try:
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as error:
        logger.error("API connectivity test failed: %s", error)
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(error),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Clay API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "model": "/v1/models/{model_id}",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }


def _responses_uses_max_completion_tokens(model_name: str) -> bool:
    normalized_model_name = model_name.lower()
    return normalized_model_name.startswith(("o1", "o3", "o4", "gpt-5"))


def _normalize_responses_input_image(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_url = item.get("image_url")
    if isinstance(image_url, str) and image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}

    if isinstance(image_url, dict):
        url_value = image_url.get("url")
        if isinstance(url_value, str) and url_value:
            image_payload: Dict[str, Any] = {"url": url_value}
            detail = image_url.get("detail")
            if isinstance(detail, str) and detail:
                image_payload["detail"] = detail
            return {"type": "image_url", "image_url": image_payload}

    return None


def _responses_input_to_chat_messages(input_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(input_payload, str):
        return [{"role": "user", "content": input_payload}]

    if isinstance(input_payload, dict):
        return _responses_dict_input_to_chat_messages(input_payload)

    if isinstance(input_payload, list):
        messages: List[Dict[str, Any]] = []
        for item in input_payload:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if isinstance(item, dict):
                messages.extend(_responses_dict_input_to_chat_messages(item))
        return messages

    return []


def _responses_dict_input_to_chat_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = item.get("role")
    if isinstance(role, str):
        content = item.get("content")
        if content is None and "text" in item:
            content = item.get("text")
        return [{"role": role, "content": _normalize_openai_chat_content(content)}]

    item_type = item.get("type")
    if item_type == "input_text":
        text = item.get("text")
        if isinstance(text, str):
            return [{"role": "user", "content": text}]
        return []

    if item_type == "input_image":
        image_part = _normalize_responses_input_image(item)
        if image_part is None:
            return []
        return [{"role": "user", "content": [image_part]}]
    if item_type == "message":
        nested_role = item.get("role", "user")
        nested_content = item.get("content", "")
        if not isinstance(nested_role, str):
            nested_role = "user"
        return [{"role": nested_role, "content": _normalize_openai_chat_content(nested_content)}]

    if item_type == "function_call_output":
        call_id = item.get("call_id") or item.get("tool_call_id")
        output_content = item.get("output")
        if call_id is None:
            return []
        return [
            {
                "role": "tool",
                "tool_call_id": str(call_id),
                "content": _coerce_text(output_content),
            }
        ]

    if item_type in {"function_call", "tool_call"}:
        function_name = item.get("name")
        call_id = item.get("call_id") or item.get("id")
        arguments = item.get("arguments", "")
        if not isinstance(function_name, str) or not function_name:
            return []
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": (
                            str(call_id) if call_id is not None else f"call_{uuid.uuid4().hex[:24]}"
                        ),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": _coerce_text(arguments),
                        },
                    }
                ],
            }
        ]

    return []


def _responses_request_to_chat_request(
    request: OpenAIResponsesRequest,
    chat_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = list(chat_messages)
    if request.instructions:
        messages = [{"role": "system", "content": request.instructions}, *messages]

    chat_request: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    }
    if request.max_output_tokens is not None:
        if _responses_uses_max_completion_tokens(request.model):
            chat_request["max_completion_tokens"] = request.max_output_tokens
        else:
            chat_request["max_tokens"] = request.max_output_tokens
    if request.tools is not None:
        chat_request["tools"] = request.tools
    if request.tool_choice is not None:
        chat_request["tool_choice"] = request.tool_choice
    if request.parallel_tool_calls is not None:
        chat_request["parallel_tool_calls"] = request.parallel_tool_calls
    if request.temperature is not None:
        chat_request["temperature"] = request.temperature
    if request.top_p is not None:
        chat_request["top_p"] = request.top_p
    if request.metadata is not None:
        chat_request["metadata"] = request.metadata

    return chat_request


def _normalize_openai_chat_content(content: Any) -> Any:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        normalized_parts: List[Any] = []
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue

            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                continue

            if item_type == "input_image":
                image_part = _normalize_responses_input_image(item)
                if image_part is not None:
                    normalized_parts.append(image_part)
                continue

            normalized_parts.append(item)

        if normalized_parts:
            if text_parts:
                normalized_parts.insert(0, {"type": "text", "text": "".join(text_parts)})
            return normalized_parts
        return "".join(text_parts)

    if isinstance(content, dict):
        text_value = content.get("text") if isinstance(content.get("text"), str) else None
        if text_value is not None:
            return text_value
        return content

    return str(content)


def _chat_completion_to_responses_payload(
    chat_payload: Dict[str, Any],
) -> Dict[str, Any]:
    choices = chat_payload.get("choices", [])
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    output_text = _extract_text_from_openai_content(message.get("content"))
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None

    output_items: List[Dict[str, Any]] = []
    if output_text or not tool_calls:
        output_items.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                    }
                ],
            }
        )

    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_info = tool_call.get("function")
            if not isinstance(function_info, dict):
                continue
            function_name = function_info.get("name")
            arguments = function_info.get("arguments", "")
            if not isinstance(function_name, str) or not function_name:
                continue

            call_id = tool_call.get("id")
            output_items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": (
                        str(call_id) if call_id is not None else f"call_{uuid.uuid4().hex[:24]}"
                    ),
                    "name": function_name,
                    "arguments": _coerce_text(arguments),
                    "status": "completed",
                }
            )

    usage = chat_payload.get("usage")
    normalized_usage = _normalize_openai_usage(usage)

    return {
        "id": chat_payload.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
        "object": "response",
        "created_at": int(datetime.now().timestamp()),
        "status": "completed",
        "model": chat_payload.get("model"),
        "output": output_items,
        "usage": normalized_usage,
    }


def _responses_stream_error_payload(error_payload: Any) -> Dict[str, Any]:
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        if not isinstance(message, str) or not message:
            message = _coerce_text(error_payload) or "Upstream provider returned an error"

        error_type = error_payload.get("type") or error_payload.get("code")
        if not isinstance(error_type, str) or not error_type:
            error_type = "server_error"

        return {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }

    return {
        "type": "error",
        "error": {
            "type": "server_error",
            "message": _coerce_text(error_payload) or "Upstream provider returned an error",
        },
    }


async def _openai_chat_stream_to_responses_events(
    openai_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    text_item_id = f"msg_{uuid.uuid4().hex[:24]}"
    text_content_index = 0
    accumulated_text = ""
    text_item_started = False
    text_content_part_started = False
    text_output_index: Optional[int] = None
    usage_payload: Dict[str, Any] = {}
    tool_state_by_index: Dict[int, Dict[str, Any]] = {}
    next_output_index = 0
    sequence_number = 0

    def _emit(event: str, payload: Dict[str, Any]) -> str:
        nonlocal sequence_number
        payload_with_sequence = dict(payload)
        payload_with_sequence["sequence_number"] = sequence_number
        sequence_number += 1
        return _sse_event(event, payload_with_sequence)

    yield _emit(
        "response.created",
        {
            "type": "response.created",
            "response": {"id": response_id},
        },
    )
    yield _emit(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "response": {"id": response_id},
        },
    )

    async for raw_line in openai_stream:
        if not raw_line.startswith("data: "):
            continue

        data = raw_line[6:].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        chunk_error = chunk.get("error")
        if chunk_error is not None:
            yield _emit("error", _responses_stream_error_payload(chunk_error))
            return

        chunk_usage = _normalize_openai_usage(chunk.get("usage"))
        if chunk_usage:
            usage_payload = chunk_usage

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue

        delta = first_choice.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}

        text = delta.get("content")
        if text is not None:
            if not isinstance(text, str):
                text = str(text)
            if text:
                accumulated_text += text
                if not text_item_started:
                    text_item_started = True
                    if text_output_index is None:
                        text_output_index = next_output_index
                        next_output_index += 1
                    yield _emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "response_id": response_id,
                            "output_index": text_output_index,
                            "item": {
                                "id": text_item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "",
                                    }
                                ],
                            },
                        },
                    )
                    if not text_content_part_started:
                        text_content_part_started = True
                        yield _emit(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "item_id": text_item_id,
                                "output_index": text_output_index,
                                "content_index": text_content_index,
                                "part": {"type": "output_text", "text": ""},
                            },
                        )
                yield _emit(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "output_index": text_output_index,
                        "item_id": text_item_id,
                        "content_index": text_content_index,
                        "delta": text,
                    },
                )

        delta_tool_calls = delta.get("tool_calls")
        if isinstance(delta_tool_calls, list):
            for tool_call in delta_tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                raw_index = tool_call.get("index", 0)
                try:
                    call_index = int(raw_index)
                except (TypeError, ValueError):
                    call_index = 0

                existing = tool_state_by_index.get(call_index)
                if existing is None:
                    item_id = f"fc_{uuid.uuid4().hex[:24]}"
                    call_id = tool_call.get("id")
                    function_value = tool_call.get("function")
                    function_info = function_value if isinstance(function_value, dict) else {}
                    name_value = function_info.get("name")
                    name = str(name_value) if name_value is not None else ""
                    existing = {
                        "item_id": item_id,
                        "call_id": (
                            str(call_id) if call_id is not None else f"call_{uuid.uuid4().hex[:24]}"
                        ),
                        "name": name,
                        "arguments": "",
                        "output_index": next_output_index,
                    }
                    next_output_index += 1
                    tool_state_by_index[call_index] = existing
                    yield _emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "response_id": response_id,
                            "output_index": existing["output_index"],
                            "item": {
                                "id": existing["item_id"],
                                "type": "function_call",
                                "call_id": existing["call_id"],
                                "name": existing["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                    )
                else:
                    function_value = tool_call.get("function")
                    function_info = function_value if isinstance(function_value, dict) else {}
                    maybe_name = function_info.get("name")
                    if isinstance(maybe_name, str) and maybe_name and not existing["name"]:
                        existing["name"] = maybe_name

                function_value = tool_call.get("function")
                function_info = function_value if isinstance(function_value, dict) else {}
                arguments_chunk = function_info.get("arguments")
                if arguments_chunk is None:
                    continue
                if not isinstance(arguments_chunk, str):
                    arguments_chunk = str(arguments_chunk)
                if not arguments_chunk:
                    continue

                existing["arguments"] += arguments_chunk
                yield _emit(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": response_id,
                        "output_index": existing["output_index"],
                        "item_id": existing["item_id"],
                        "call_id": existing["call_id"],
                        "delta": arguments_chunk,
                    },
                )

        finish_reason = first_choice.get("finish_reason")
        if finish_reason in {"tool_calls", "function_call"}:
            for call_index in sorted(tool_state_by_index.keys()):
                state = tool_state_by_index[call_index]
                yield _emit(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": response_id,
                        "output_index": state["output_index"],
                        "item_id": state["item_id"],
                        "call_id": state["call_id"],
                        "name": state["name"],
                        "arguments": state["arguments"],
                    },
                )
                yield _emit(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": state["output_index"],
                        "item": {
                            "id": state["item_id"],
                            "type": "function_call",
                            "call_id": state["call_id"],
                            "name": state["name"],
                            "arguments": state["arguments"],
                            "status": "completed",
                        },
                    },
                )
            tool_state_by_index.clear()

    if text_item_started:
        yield _emit(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "response_id": response_id,
                "output_index": text_output_index,
                "item_id": text_item_id,
                "content_index": text_content_index,
                "text": accumulated_text,
            },
        )
        yield _emit(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": text_item_id,
                "output_index": text_output_index,
                "content_index": text_content_index,
                "part": {"type": "output_text", "text": accumulated_text},
            },
        )
        yield _emit(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": text_output_index,
                "item": {
                    "id": text_item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": accumulated_text,
                        }
                    ],
                },
            },
        )

    for call_index in sorted(tool_state_by_index.keys()):
        state = tool_state_by_index[call_index]
        yield _emit(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "response_id": response_id,
                "output_index": state["output_index"],
                "item_id": state["item_id"],
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"],
            },
        )
        yield _emit(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": state["output_index"],
                "item": {
                    "id": state["item_id"],
                    "type": "function_call",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                    "status": "completed",
                },
            },
        )

    yield _emit(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "status": "completed",
                "usage": usage_payload,
            },
        },
    )


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize_openai_usage(usage: Any) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        return {}

    normalized: Dict[str, Any] = dict(usage)

    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_int(usage.get("completion_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))

    if prompt_tokens is not None:
        normalized["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        normalized["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
    elif prompt_tokens is not None:
        normalized.setdefault("input_tokens", prompt_tokens)
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
    elif completion_tokens is not None:
        normalized.setdefault("output_tokens", completion_tokens)
    if total_tokens is not None:
        normalized.setdefault("total_tokens", total_tokens)

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        coerced_prompt_details: Dict[str, Any] = dict(prompt_details)
        for key, value in prompt_details.items():
            coerced_value = _coerce_int(value)
            if coerced_value is not None:
                coerced_prompt_details[key] = coerced_value
        normalized["prompt_tokens_details"] = coerced_prompt_details
        normalized["input_tokens_details"] = coerced_prompt_details

    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        coerced_completion_details: Dict[str, Any] = dict(completion_details)
        for key, value in completion_details.items():
            coerced_value = _coerce_int(value)
            if coerced_value is not None:
                coerced_completion_details[key] = coerced_value
        normalized["completion_tokens_details"] = coerced_completion_details
        normalized["output_tokens_details"] = coerced_completion_details

    return normalized


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


async def _format_openai_sse_stream(
    openai_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    async for line in openai_stream:
        trimmed = line.strip()
        if not trimmed:
            continue
        yield f"{trimmed}\n\n"


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _estimate_request_input_tokens(request: ClaudeTokenCountRequestModel) -> int:
    openai_model = model_manager.map_claude_model_to_openai(request.model)
    total_tokens = 0

    if request.system:
        if isinstance(request.system, str):
            total_tokens += _estimate_tokens_for_text(request.system, openai_model)
        else:
            for system_block in request.system:
                total_tokens += _estimate_tokens_for_text(system_block.text, openai_model)

    for message in request.messages:
        total_tokens += _estimate_message_tokens(message.content, openai_model)

    if request.tools:
        total_tokens += _estimate_tools_tokens(request.tools, openai_model)
    if request.tool_choice:
        total_tokens += _estimate_tokens_for_text(
            json.dumps(request.tool_choice.model_dump(exclude_none=True), ensure_ascii=False),
            openai_model,
        )
    if request.thinking:
        total_tokens += _estimate_tokens_for_text(
            json.dumps(request.thinking.model_dump(exclude_none=True), ensure_ascii=False),
            openai_model,
        )

    return max(1, total_tokens)


def _estimate_message_tokens(content: Any, model_name: str) -> int:
    if isinstance(content, str):
        return _estimate_tokens_for_text(content, model_name)

    if not isinstance(content, list):
        return _estimate_tokens_for_text(str(content), model_name)

    message_tokens = 0
    for content_block in content:
        message_tokens += _estimate_content_block_tokens(content_block, model_name)
    return message_tokens


def _estimate_content_block_tokens(content_block: Any, model_name: str) -> int:
    block_type = getattr(content_block, "type", None)

    if block_type == "text":
        return _estimate_tokens_for_text(getattr(content_block, "text", ""), model_name)

    if block_type == "thinking":
        return _estimate_tokens_for_text(getattr(content_block, "thinking", ""), model_name)

    if block_type == "redacted_thinking":
        return _estimate_tokens_for_text(getattr(content_block, "data", ""), model_name)

    if block_type == "document":
        source = getattr(content_block, "source", None)
        if source is not None and getattr(source, "type", None) == "text":
            return _estimate_tokens_for_text(getattr(source, "text", ""), model_name)
        source_data = ""
        if source is not None and hasattr(source, "data"):
            source_data = str(getattr(source, "data", ""))
        title = getattr(content_block, "title", "") or ""
        context = getattr(content_block, "context", "") or ""
        estimated = max(1, len(source_data) // 6)
        estimated += _estimate_tokens_for_text(f"{title}\n{context}", model_name)
        return estimated

    if block_type == "image":
        source = getattr(content_block, "source", None)
        source_data = ""
        if source is not None and hasattr(source, "data"):
            source_data = str(getattr(source, "data", ""))
        return max(85, len(source_data) // 48)

    if block_type == "tool_result":
        normalized = parse_tool_result_content(getattr(content_block, "content", ""))
        return _estimate_tokens_for_text(normalized, model_name)

    if block_type == "tool_use":
        tool_name = getattr(content_block, "name", "")
        tool_input = getattr(content_block, "input", {})
        tool_payload = f"{tool_name}\n{json.dumps(tool_input, ensure_ascii=False)}"
        return _estimate_tokens_for_text(tool_payload, model_name)

    return _estimate_tokens_for_text(str(content_block), model_name)


def _estimate_tools_tokens(tools: List[Any], model_name: str) -> int:
    serialized: List[Dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            serialized.append(tool.model_dump(exclude_none=True))
        elif isinstance(tool, dict):
            serialized.append(tool)
        else:
            serialized.append({"value": str(tool)})

    payload = json.dumps(serialized, ensure_ascii=False)
    return _estimate_tokens_for_text(payload, model_name)


def _estimate_tokens_for_text(text: str, model_name: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0

    tiktoken_count = _token_count_with_tiktoken(normalized, model_name)
    if tiktoken_count is not None:
        return tiktoken_count

    return max(1, len(normalized) // 4)


def _token_count_with_tiktoken(text: str, model_name: str) -> Optional[int]:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    try:
        return len(encoding.encode(text))
    except Exception:
        return None


def _extract_text_from_openai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return str(content)
