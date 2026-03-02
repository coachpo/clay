import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.conversion.request_converter import (
    convert_claude_to_openai,
    parse_tool_result_content,
)
from app.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from app.core.client import OpenAIClient
from app.core.config import config
from app.core.logging import logger
from app.core.model_manager import model_manager
from app.models.claude import (
    ClaudeMessagesRequestModel,
    ClaudeTokenCountRequestModel,
    parse_claude_messages_request,
    parse_claude_token_count_request,
)

router = APIRouter()

openai_api_key = config.openai_api_key or ""
openai_client = OpenAIClient(
    openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    config.max_retries,
    api_version=config.azure_api_version,
)


MAX_ANTHROPIC_REQUEST_BYTES = 32 * 1024 * 1024
ADAPTIVE_THINKING_MODEL_PREFIXES = ("claude-opus-4-6", "claude-sonnet-4-6")


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


def _model_supports_adaptive_thinking(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith(ADAPTIVE_THINKING_MODEL_PREFIXES)


def _validate_thinking_model_compatibility(
    request: ClaudeMessagesRequestModel,
    request_id: str,
) -> None:
    if request.thinking is None or request.thinking.type != "adaptive":
        return
    if _model_supports_adaptive_thinking(request.model):
        return
    raise HTTPException(
        status_code=400,
        detail=_anthropic_error_response(
            message=f"Model '{request.model}' does not support adaptive thinking.",
            error_type="invalid_request_error",
            request_id=request_id,
        ),
    )


def _build_openai_request_from_claude(
    claude_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
    openai_request = convert_claude_to_openai(claude_request, model_manager)

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
    if claude_request.context_management is not None:
        extension_fields["context_management"] = claude_request.context_management.model_dump(
            exclude_none=True
        )
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
    content_type: Optional[str] = Header(None, alias="content-type"),
) -> None:
    _enforce_request_size_limit(request)

    _resolve_requested_anthropic_version(
        anthropic_version=anthropic_version,
    )
    if content_type and not content_type.lower().startswith("application/json"):
        raise _http_exception_from_message(
            status_code=400,
            message="Unsupported Content-Type. Expected application/json.",
        )

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
        _validate_thinking_model_compatibility(request, request_id)
        logger.debug(
            "Processing Claude request: model=%s, stream=%s",
            request.model,
            request.stream,
        )

        responses_request = _build_openai_request_from_claude(request)

        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            responses_stream = openai_client.create_response_stream(responses_request, request_id)
            return StreamingResponse(
                convert_openai_streaming_to_claude_with_cancellation(
                    responses_stream,
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

        responses_response = await _create_response_with_disconnect_cancellation(
            responses_request,
            http_request,
            request_id,
        )
        claude_response = convert_openai_to_claude_response(responses_response, request)
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


async def _create_response_with_disconnect_cancellation(
    responses_request: Dict[str, Any],
    http_request: Request,
    request_id: str,
) -> Dict[str, Any]:
    completion_task = asyncio.create_task(
        openai_client.create_response(responses_request, request_id)
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
async def chat_completions_removed() -> Any:
    request_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=404,
        content=_openai_error_response(
            "Endpoint removed. Use POST /v1/messages for generation.",
            error_type="invalid_request_error",
            code="not_found",
        ),
        headers=_request_id_headers(request_id),
    )


@router.post("/v1/responses")
async def responses_removed() -> Any:
    request_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=404,
        content=_openai_error_response(
            "Endpoint removed. Use POST /v1/messages for generation.",
            error_type="invalid_request_error",
            code="not_found",
        ),
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
        test_response = await openai_client.create_response(
            {
                "model": config.small_model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    }
                ],
                "max_output_tokens": 5,
                "store": False,
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
            "models": "/v1/models",
            "model": "/v1/models/{model_id}",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }


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
    if request.context_management:
        total_tokens += _estimate_tokens_for_text(
            json.dumps(
                request.context_management.model_dump(exclude_none=True), ensure_ascii=False
            ),
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
