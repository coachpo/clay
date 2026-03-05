"""Integration tests for strict Anthropic and OpenAI compatibility over the proxy."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from openai._exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import app.api.endpoints as api_endpoints
from app.api.endpoints import _is_supported_openai_model_id
from app.conversion.request_converter import convert_claude_to_openai, parse_tool_result_content
from app.conversion.response_converter import (
    convert_openai_streaming_to_claude,
    convert_openai_to_claude_response,
)
from app.core.client import OpenAIClient
from app.core.config import config
from app.core.logging import logger
from app.core.model_manager import model_manager
from app.main import app
from app.models.claude import parse_claude_messages_request

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CLIENT_API_KEY = os.getenv("ANTHROPIC_API_KEY", "test-key")

ANTHROPIC_HEADERS = {
    "anthropic-version": config.anthropic_default_version,
    "x-api-key": CLIENT_API_KEY,
}

OPENAI_HEADERS = {
    "authorization": f"Bearer {CLIENT_API_KEY}",
    "content-type": "application/json",
}

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", config.middle_model)
ANTHROPIC_STREAM_MODEL = os.getenv("ANTHROPIC_STREAM_MODEL", config.small_model)


def _extract_model_ids_from_models_payload(payload: Dict[str, Any]) -> List[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return model_ids


async def _resolve_runtime_models() -> None:
    global ANTHROPIC_MODEL, ANTHROPIC_STREAM_MODEL
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{BASE_URL}/v1/models", headers=OPENAI_HEADERS)
        if response.status_code != 200:
            return
        model_ids = _extract_model_ids_from_models_payload(response.json())
        if not model_ids:
            return
        selected = model_ids[0]
        ANTHROPIC_MODEL = selected
        ANTHROPIC_STREAM_MODEL = selected
    except Exception:
        return


def _build_claude_request_model(payload: Dict[str, Any]) -> Any:
    return parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )


def _pretty(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


async def _post_messages(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    request_headers = dict(ANTHROPIC_HEADERS)
    if headers:
        request_headers.update(headers)
    return await client.post(f"{BASE_URL}/v1/messages", json=payload, headers=request_headers)


class _StubOpenAIClientForMessages:
    def __init__(self) -> None:
        self.last_request: Optional[Dict[str, Any]] = None

    async def create_response(
        self, responses_request: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        self.last_request = dict(responses_request)
        return {
            "id": request_id or "resp_stub",
            "output": [
                {
                    "type": "message",
                    "id": "msg_stub",
                    "content": [{"type": "output_text", "text": "stubbed"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    def cancel_request(self, _: str) -> bool:
        return True

    def classify_openai_error(self, message: str) -> str:
        return message


async def _post_messages_in_process(
    payload: Dict[str, Any],
    query_string: str = "",
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    path = "/v1/messages"
    if query_string:
        path = f"{path}?{query_string}"
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(path, json=payload, headers=ANTHROPIC_HEADERS)


async def _post_count_tokens(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    request_headers = dict(ANTHROPIC_HEADERS)
    if headers:
        request_headers.update(headers)
    return await client.post(
        f"{BASE_URL}/v1/messages/count_tokens",
        json=payload,
        headers=request_headers,
    )


async def _post_openai_chat(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    request_headers = dict(OPENAI_HEADERS)
    if headers:
        request_headers.update(headers)
    return await client.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers=request_headers,
    )


async def _post_openai_responses(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    request_headers = dict(OPENAI_HEADERS)
    if headers:
        request_headers.update(headers)
    return await client.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        headers=request_headers,
    )


def _assert_request_id_headers(response: httpx.Response) -> None:
    request_id = response.headers.get("request-id")
    x_request_id = response.headers.get("x-request-id")
    assert isinstance(request_id, str) and request_id, response.headers
    assert request_id == x_request_id, response.headers


def _assert_anthropic_error_shape(response: httpx.Response, expected_status: int) -> Dict[str, Any]:
    assert response.status_code == expected_status, response.text
    payload = response.json()
    assert payload.get("type") == "error", payload
    assert isinstance(payload.get("error"), dict), payload
    assert isinstance(payload["error"].get("type"), str), payload
    assert isinstance(payload["error"].get("message"), str), payload
    assert isinstance(payload.get("request_id"), str), payload
    _assert_request_id_headers(response)
    return payload


def _assert_openai_error_shape(response: httpx.Response, expected_status: int) -> Dict[str, Any]:
    assert response.status_code == expected_status, response.text
    payload = response.json()
    error = payload.get("error")
    assert isinstance(error, dict), payload
    assert isinstance(error.get("message"), str), payload
    assert isinstance(error.get("type"), str), payload
    _assert_request_id_headers(response)
    return payload


def _is_upstream_unavailable(response: httpx.Response) -> bool:
    if response.status_code in {502, 503, 504}:
        return True

    if response.status_code >= 500:
        text = response.text.lower()
        if "bad gateway" in text or "<!doctype html" in text:
            return True

    return False


def _extract_stream_events(raw_lines: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current_event: Optional[str] = None
    current_data: Optional[str] = None

    for raw_line in raw_lines:
        line = raw_line.strip()
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
        elif line.startswith("data: "):
            current_data = line[len("data: ") :]
        elif line == "" and current_event and current_data:
            events.append({"event": current_event, "data": json.loads(current_data)})
            current_event = None
            current_data = None

    if current_event and current_data:
        events.append({"event": current_event, "data": json.loads(current_data)})

    return events


async def _iter_lines(lines: List[str]) -> Any:
    for line in lines:
        yield line


async def _collect_sse_events(
    generator: Any,
    request_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    request_model = parse_claude_messages_request(
        request_payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    output_lines: List[str] = []
    async for line in convert_openai_streaming_to_claude(generator, request_model, logger):
        output_lines.append(line)

    split_lines: List[str] = []
    for payload_line in output_lines:
        split_lines.extend(payload_line.splitlines())
        split_lines.append("")

    return _extract_stream_events(split_lines)


async def _provider_is_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(f"{BASE_URL}/test-connection")
        return response.status_code == 200
    except Exception:
        return False


async def _require_provider_or_skip(test_name: str) -> bool:
    if await _provider_is_reachable():
        return True

    print(f"- {test_name} skipped (upstream provider unavailable)")
    return False


def _bad_request_error(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://example.com/v1/responses")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(
        message=message,
        response=response,
        body={"error": {"message": message}},
    )


def _api_status_error(
    message: str,
    status_code: int,
    *,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/v1/responses")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        headers=headers or {"content-type": "application/json"},
    )
    return APIStatusError(
        message=message,
        response=response,
        body=body or {"error": {"message": message}},
    )


def _api_response_validation_error(
    message: str,
    *,
    status_code: int = 200,
    content_type: str = "text/html; charset=utf-8",
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
    text: Optional[str] = None,
    url: str = "https://example.com/v1/responses",
) -> APIResponseValidationError:
    request = httpx.Request("POST", url)
    response_kwargs: Dict[str, Any] = {
        "status_code": status_code,
        "request": request,
        "headers": headers or {"content-type": content_type},
    }

    if text is not None:
        response_kwargs["text"] = text
    elif body is not None and not isinstance(body, (dict, list)):
        response_kwargs["text"] = str(body)
    elif body is None:
        response_kwargs["text"] = ""

    response = httpx.Response(**response_kwargs)
    return APIResponseValidationError(response=response, body=body, message=message)


def test_protocol_error_classification_excludes_transport_failures() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    request = httpx.Request("POST", "https://example.com/v1/responses")

    connection_error = APIConnectionError(request=request)
    timeout_error = APITimeoutError(request=request)

    assert client._is_protocol_error_api_error(connection_error) is False
    assert client._is_protocol_error_api_error(timeout_error) is False


def test_openai_client_normalizes_root_base_url_to_v1() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://api.duckcoding.ai")
    assert client.base_url == "https://api.duckcoding.ai/v1", client.base_url


def test_openai_client_preserves_explicit_versioned_base_url() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://codex-api.packycode.com/v1")
    assert client.base_url == "https://codex-api.packycode.com/v1", client.base_url


def test_openai_client_keeps_raw_base_url_for_azure_mode() -> None:
    client = OpenAIClient(
        api_key="sk-test",
        base_url="https://example-resource.openai.azure.com",
        api_version="2024-10-21",
    )
    assert client.base_url == "https://example-resource.openai.azure.com", client.base_url


async def test_missing_anthropic_version_header() -> None:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": CLIENT_API_KEY},
        )
        _assert_anthropic_error_shape(response, expected_status=400)


async def test_retries_without_metadata_on_unsupported_parameter_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    unsupported_metadata_error = _bad_request_error("Unsupported parameter: metadata")

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise unsupported_metadata_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "metadata": {"user_id": "test-user"},
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "metadata" in calls[0]
    assert "metadata" not in calls[1]


async def test_retries_without_context_management_on_unsupported_parameter_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    unsupported_context_management_error = _bad_request_error(
        "Unsupported parameter: context_management"
    )

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise unsupported_context_management_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "context_management" in calls[0]
    assert "context_management" not in calls[1]


async def test_retries_without_extra_body_on_unsupported_parameter_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    unsupported_extra_body_error = _bad_request_error("Unsupported parameter: extra_body")

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise unsupported_extra_body_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "extra_body": {
            "proxy_metadata": {"anthropic_extensions": {"thinking": {"type": "adaptive"}}}
        },
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "extra_body" in calls[0]
    assert "extra_body" not in calls[1]


async def test_retries_once_without_all_optional_fields_when_any_optional_field_is_rejected() -> (
    None
):
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    unsupported_metadata_error = _bad_request_error("Unsupported parameter: metadata")

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise unsupported_metadata_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "metadata": {"user_id": "test-user"},
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "metadata" in calls[0]
    assert "context_management" in calls[0]
    assert "metadata" not in calls[1]
    assert "context_management" not in calls[1]


async def test_does_not_retry_for_other_bad_request_errors() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    unrelated_bad_request_error = _bad_request_error("Request body is invalid")

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        raise unrelated_bad_request_error

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "metadata": {"user_id": "test-user"},
    }

    try:
        await client._create_with_metadata_fallback(request)
    except BadRequestError:
        pass
    else:
        raise AssertionError("Expected BadRequestError for unrelated request validation errors")

    assert len(calls) == 1


async def test_retries_without_context_management_on_retryable_server_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    upstream_502_error = _api_status_error("Bad gateway", status_code=502)

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise upstream_502_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "context_management" in calls[0]
    assert "context_management" not in calls[1]


async def test_retries_without_proxy_metadata_context_management_on_retryable_server_error() -> (
    None
):
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    upstream_502_error = _api_status_error("Bad gateway", status_code=502)

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise upstream_502_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
        "extra_body": {
            "proxy_metadata": {
                "anthropic_extensions": {
                    "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
                },
                "original_anthropic_request": {
                    "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
                },
            }
        },
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "context_management" in calls[0]
    assert "context_management" not in calls[1]

    assert "extra_body" not in calls[1]


async def test_retries_without_extra_body_on_retryable_server_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    upstream_502_error = _api_status_error("Bad gateway", status_code=502)

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise upstream_502_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "extra_body": {
            "proxy_metadata": {"anthropic_extensions": {"thinking": {"type": "adaptive"}}}
        },
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "extra_body" in calls[0]
    assert "extra_body" not in calls[1]


async def test_retries_without_optional_fields_on_protocol_api_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()
    protocol_error = _api_status_error(
        "bad response body",
        status_code=500,
        body={
            "error": {
                "message": "invalid character 'e' looking for beginning of value",
                "type": "bad_response_body",
                "code": "bad_response_body",
            }
        },
    )

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise protocol_error
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "metadata": {"user_id": "test-user"},
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
        "extra_body": {
            "proxy_metadata": {"anthropic_extensions": {"thinking": {"type": "adaptive"}}}
        },
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "metadata" in calls[0]
    assert "context_management" in calls[0]
    assert "extra_body" in calls[0]
    assert "metadata" not in calls[1]
    assert "context_management" not in calls[1]
    assert "extra_body" not in calls[1]


async def test_retries_once_on_json_decode_protocol_error_without_optional_fields() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    sentinel = object()

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise json.JSONDecodeError("Expecting value", "e", 0)
        return sentinel

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "metadata": {"user_id": "test-user"},
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    }

    result = await client._create_with_metadata_fallback(request)
    assert result is sentinel
    assert len(calls) == 2
    assert "metadata" in calls[0]
    assert "context_management" in calls[0]
    assert "metadata" not in calls[1]
    assert "context_management" not in calls[1]


def test_retryable_server_status_includes_cloudflare_and_529() -> None:
    for status in [500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 529]:
        error = _api_status_error("retryable", status)
        assert OpenAIClient._is_retryable_server_status(error) is True

    for status in [400, 401, 403, 404, 408, 409, 422, 429]:
        error = _api_status_error("not-retryable", status)
        assert OpenAIClient._is_retryable_server_status(error) is False


def test_normalize_error_status_code_maps_non_error_to_502() -> None:
    assert OpenAIClient._normalize_error_status_code(200) == 502
    assert OpenAIClient._normalize_error_status_code(399) == 502
    assert OpenAIClient._normalize_error_status_code(None) == 502
    assert OpenAIClient._normalize_error_status_code(429) == 429


def test_content_type_json_detection() -> None:
    assert OpenAIClient._content_type_is_json("application/json") is True
    assert OpenAIClient._content_type_is_json("application/json; charset=utf-8") is True
    assert OpenAIClient._content_type_is_json("application/problem+json") is True
    assert OpenAIClient._content_type_is_json("text/html") is False
    assert OpenAIClient._content_type_is_json("text/event-stream") is False


def test_html_error_payload_signature_detection() -> None:
    assert (
        OpenAIClient._looks_like_html_error_payload("<!doctype html><html>challenge</html>") is True
    )
    assert OpenAIClient._looks_like_html_error_payload("Cloudflare Ray ID: 12345") is True
    assert OpenAIClient._looks_like_html_error_payload("Error 1020 Access denied") is True
    assert OpenAIClient._looks_like_html_error_payload('{"ok": true}') is False


def test_safe_payload_preview_normalizes_and_clips() -> None:
    preview = OpenAIClient._safe_payload_preview(" line1\nline2\tline3 ", max_chars=10)
    assert preview == "line1 line"
    dict_preview = OpenAIClient._safe_payload_preview({"a": "b"})
    assert '"a": "b"' in dict_preview


def test_build_upstream_protocol_error_uses_status_502_for_fake_200() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    error = _api_response_validation_error(
        "Response payload is not valid JSON",
        status_code=200,
        content_type="text/html; charset=utf-8",
        body="<!doctype html><html><title>502 Bad Gateway</title></html>",
        headers={
            "content-type": "text/html; charset=utf-8",
            "cf-mitigated": "challenge",
            "cf-ray": "abcd1234",
        },
    )

    http_error = client._build_upstream_protocol_error(error)
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 502
    detail = str(http_error.detail)
    assert "Upstream protocol error" in detail
    assert "unexpected content-type" in detail
    assert "cf-mitigated=challenge" in detail
    assert "cf-ray=abcd1234" in detail
    assert "Body preview" in detail


def test_build_upstream_protocol_error_keeps_upstream_error_status() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    error = _api_response_validation_error(
        "Response schema mismatch",
        status_code=503,
        content_type="application/json",
        body={"error": "upstream unavailable"},
    )

    http_error = client._build_upstream_protocol_error(error)
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 503
    assert "Upstream protocol error" in str(http_error.detail)


async def test_create_response_maps_protocol_validation_error_to_http_502() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    protocol_error = _api_response_validation_error(
        "Invalid JSON response from upstream",
        status_code=200,
        content_type="text/html",
        body="<!doctype html><html>Bad gateway</html>",
    )

    async def fake_create(**_: Any) -> Any:
        raise protocol_error

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }

    try:
        await client.create_response(request)
    except HTTPException as error:
        assert error.status_code == 502, error
        assert "Upstream protocol error" in str(error.detail), error
    else:
        raise AssertionError("Expected HTTPException(502) for protocol validation error")


async def test_create_response_stream_maps_protocol_validation_error_to_http_502() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    protocol_error = _api_response_validation_error(
        "Invalid stream envelope",
        status_code=200,
        content_type="text/html",
        body="<!doctype html><html>challenge</html>",
        headers={"content-type": "text/html", "cf-mitigated": "challenge"},
    )

    async def fake_create(**_: Any) -> Any:
        raise protocol_error

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "stream": True,
    }

    stream = client.create_response_stream(request)
    try:
        await anext(stream)
    except HTTPException as error:
        assert error.status_code == 502, error
        assert "Upstream protocol error" in str(error.detail), error
    else:
        raise AssertionError("Expected HTTPException(502) for streaming protocol validation error")


async def test_create_response_maps_json_decode_error_to_http_502() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")

    async def fake_create(**_: Any) -> Any:
        raise json.JSONDecodeError("Expecting value", "e", 0)

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }

    try:
        await client.create_response(request)
    except HTTPException as error:
        assert error.status_code == 502, error
        assert "Upstream protocol error" in str(error.detail), error
        assert "Parse failure" in str(error.detail), error
    else:
        raise AssertionError("Expected HTTPException(502) for JSON decode protocol error")


async def test_create_response_stream_maps_json_decode_error_to_http_502() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")

    async def fake_create(**_: Any) -> Any:
        raise json.JSONDecodeError("Expecting value", "e", 0)

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "stream": True,
    }

    stream = client.create_response_stream(request)
    try:
        await anext(stream)
    except HTTPException as error:
        assert error.status_code == 502, error
        assert "Upstream protocol error" in str(error.detail), error
        assert "Parse failure" in str(error.detail), error
    else:
        raise AssertionError("Expected HTTPException(502) for stream JSON decode protocol error")


async def test_create_response_maps_bad_response_body_api_error_to_protocol_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    protocol_error = _api_status_error(
        "bad response body",
        status_code=500,
        body={
            "error": {
                "message": "invalid character 'e' looking for beginning of value",
                "type": "bad_response_body",
                "code": "bad_response_body",
            }
        },
    )

    async def fake_create(**_: Any) -> Any:
        raise protocol_error

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    request = {
        "model": "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }

    try:
        await client.create_response(request)
    except HTTPException as error:
        assert error.status_code == 500, error
        assert "Upstream protocol error" in str(error.detail), error
    else:
        raise AssertionError(
            "Expected protocol-style HTTPException for bad_response_body API error"
        )


async def test_does_not_retry_for_non_retryable_api_status_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: List[Dict[str, Any]] = []
    upstream_429_error = _api_status_error("Too many requests", status_code=429)

    async def fake_create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        raise upstream_429_error

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    }

    try:
        await client._create_with_metadata_fallback(request)
    except APIStatusError:
        pass
    else:
        raise AssertionError("Expected APIStatusError(429) to bypass optional-field fallback")

    assert len(calls) == 1


async def test_lightweight_client_cancellation_for_non_stream_response() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    request_id = "req_cancel_lightweight"

    async def fake_create(**_: Any) -> Any:
        await asyncio.sleep(10)
        return SimpleNamespace(model_dump=lambda: {"id": "resp_should_not_complete"})

    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    request = {
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }

    response_task = asyncio.create_task(client.create_response(request, request_id=request_id))
    await asyncio.sleep(0.05)
    assert client.cancel_request(request_id) is True

    try:
        await response_task
    except HTTPException as error:
        assert error.status_code == 499, error
    else:
        raise AssertionError("Expected HTTPException(499) for cancelled request")

    assert request_id not in client.active_requests, client.active_requests


async def test_unknown_fields_rejected_with_anthropic_error_shape() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_messages(
            client,
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
                "unknown_field": True,
            },
        )
        _assert_anthropic_error_shape(response, expected_status=400)


async def test_invalid_role_content_rejected_with_anthropic_error_shape() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_messages(
            client,
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_123",
                                "content": "invalid in assistant role",
                            }
                        ],
                    }
                ],
            },
        )
        _assert_anthropic_error_shape(response, expected_status=400)


async def test_anthropic_large_request_returns_413() -> None:
    async with httpx.AsyncClient() as client:
        huge_text = "x" * (33 * 1024 * 1024)
        response = await _post_messages(
            client,
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": huge_text}],
            },
        )
        payload = _assert_anthropic_error_shape(response, expected_status=413)
        assert payload["error"]["type"] == "request_too_large", payload


async def test_token_count_endpoint_header_and_response_shape() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_count_tokens(
            client,
            {
                "model": ANTHROPIC_MODEL,
                "messages": [{"role": "user", "content": "Count this"}],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert isinstance(payload.get("input_tokens"), int), payload
        assert payload["input_tokens"] > 0, payload
        _assert_request_id_headers(response)


async def test_token_count_includes_tools_and_thinking() -> None:
    async with httpx.AsyncClient() as client:
        payload = {
            "model": ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": "Count tools and thinking"}],
            "tools": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"type": "auto"},
        }
        response = await _post_count_tokens(client, payload)
        assert response.status_code == 200, response.text
        value_with_tools = response.json().get("input_tokens")
        assert isinstance(value_with_tools, int) and value_with_tools > 0, response.text

        payload_without_tools = {
            "model": payload["model"],
            "messages": payload["messages"],
        }
        baseline = await _post_count_tokens(client, payload_without_tools)
        assert baseline.status_code == 200, baseline.text
        baseline_tokens = baseline.json().get("input_tokens")
        assert isinstance(baseline_tokens, int), baseline.text
        assert value_with_tools >= baseline_tokens, (value_with_tools, baseline_tokens)


async def test_token_count_rejects_web_search_conflicting_domain_filters() -> None:
    async with httpx.AsyncClient() as client:
        payload = {
            "model": ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": "Count web search"}],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["untrusted.example"],
                }
            ],
        }
        response = await _post_count_tokens(client, payload)
        _assert_anthropic_error_shape(response, expected_status=400)


async def test_messages_accepts_adaptive_thinking_for_claude_sonnet_4_6() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use adaptive thinking"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
            query_string="beta=true",
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "xhigh", stub_client.last_request


async def test_messages_maps_low_effort_to_openai_medium() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use low effort"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "medium", stub_client.last_request


async def test_messages_accepts_adaptive_thinking_for_claude_opus_4_6() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use adaptive thinking"}],
                "thinking": {"type": "adaptive"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "reasoning" not in stub_client.last_request, stub_client.last_request


async def test_messages_omitted_temperature_is_not_forwarded() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "No explicit temperature"}],
                "thinking": {"type": "adaptive"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "temperature" not in stub_client.last_request, stub_client.last_request


async def test_messages_explicit_temperature_is_forwarded() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Explicit temperature"}],
                "thinking": {"type": "disabled"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert stub_client.last_request.get("temperature") == 0.2, stub_client.last_request
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "none", stub_client.last_request


async def test_messages_explicit_temperature_requires_thinking_disabled() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Drop temperature without explicit disable"}],
                "thinking": {"type": "adaptive"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "temperature" not in stub_client.last_request, stub_client.last_request


async def test_messages_explicit_temperature_scales_to_x2_when_enabled() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    original_scale_flag = config.anthropic_temperature_scale_to_openai_x2
    api_endpoints.openai_client = stub_client
    config.anthropic_temperature_scale_to_openai_x2 = True
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Scaled temperature"}],
                "thinking": {"type": "disabled"},
            }
        )
    finally:
        config.anthropic_temperature_scale_to_openai_x2 = original_scale_flag
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert stub_client.last_request.get("temperature") == 0.4, stub_client.last_request
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "none", stub_client.last_request


async def test_messages_drops_sampling_fields_when_reasoning_conflicts_by_default() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    original_mode = config.openai_gpt5_sampling_reasoning_compat_mode
    api_endpoints.openai_client = stub_client
    config.openai_gpt5_sampling_reasoning_compat_mode = "drop_sampling"
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "top_p": 0.9,
                "messages": [{"role": "user", "content": "Conflict for drop mode"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"},
            }
        )
    finally:
        config.openai_gpt5_sampling_reasoning_compat_mode = original_mode
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "temperature" not in stub_client.last_request, stub_client.last_request
    assert "top_p" not in stub_client.last_request, stub_client.last_request
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "xhigh", stub_client.last_request


async def test_messages_force_reasoning_none_when_sampling_conflicts() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    original_mode = config.openai_gpt5_sampling_reasoning_compat_mode
    api_endpoints.openai_client = stub_client
    config.openai_gpt5_sampling_reasoning_compat_mode = "force_reasoning_none"
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Conflict for force-none mode"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"},
            }
        )
    finally:
        config.openai_gpt5_sampling_reasoning_compat_mode = original_mode
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert stub_client.last_request.get("temperature") == 0.2, stub_client.last_request
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "none", stub_client.last_request


async def test_messages_strict_error_when_sampling_conflicts() -> None:
    original_mode = config.openai_gpt5_sampling_reasoning_compat_mode
    config.openai_gpt5_sampling_reasoning_compat_mode = "strict_error"
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Conflict for strict mode"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"},
            }
        )
    finally:
        config.openai_gpt5_sampling_reasoning_compat_mode = original_mode

    payload = _assert_anthropic_error_shape(response, expected_status=400)
    message = payload["error"].get("message", "")
    assert "Sampling fields temperature/top_p require thinking.type='disabled'" in message, payload


async def test_messages_non_gpt5_model_applies_sampling_compatibility() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    original_mode = config.openai_gpt5_sampling_reasoning_compat_mode
    api_endpoints.openai_client = stub_client
    config.openai_gpt5_sampling_reasoning_compat_mode = "drop_sampling"
    try:
        response = await _post_messages_in_process(
            {
                "model": "gpt-4o",
                "max_tokens": 64,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "Non-gpt5 model"}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            }
        )
    finally:
        config.openai_gpt5_sampling_reasoning_compat_mode = original_mode
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "temperature" not in stub_client.last_request, stub_client.last_request
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "high", stub_client.last_request


async def test_messages_maps_max_effort_to_openai_xhigh() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use max effort"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "max"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "xhigh", stub_client.last_request


async def test_messages_enabled_thinking_with_budget_tokens_still_works_for_sonnet_4_5() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use classic thinking"}],
                "thinking": {"type": "enabled", "budget_tokens": 8192},
                "output_config": {"effort": "medium"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "high", stub_client.last_request


async def test_messages_enabled_thinking_budget_tokens_maps_to_xhigh_without_output_config() -> (
    None
):
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use classic thinking"}],
                "thinking": {"type": "enabled", "budget_tokens": 8192},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "xhigh", stub_client.last_request


async def test_messages_output_config_without_effort_falls_back_to_thinking_budget() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use classic thinking"}],
                "thinking": {"type": "enabled", "budget_tokens": 8192},
                "output_config": {},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    reasoning = stub_client.last_request.get("reasoning")
    assert isinstance(reasoning, dict), stub_client.last_request
    assert reasoning.get("effort") == "xhigh", stub_client.last_request


async def test_messages_enabled_thinking_without_budget_turns_reasoning_off() -> None:
    stub_client = _StubOpenAIClientForMessages()
    original_client = api_endpoints.openai_client
    api_endpoints.openai_client = stub_client
    try:
        response = await _post_messages_in_process(
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Use classic thinking"}],
                "thinking": {"type": "enabled"},
            }
        )
    finally:
        api_endpoints.openai_client = original_client

    assert 200 <= response.status_code < 300, response.text
    _assert_request_id_headers(response)
    assert stub_client.last_request is not None
    assert "reasoning" not in stub_client.last_request, stub_client.last_request


async def test_messages_rejects_adaptive_thinking_for_older_models() -> None:
    response = await _post_messages_in_process(
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use adaptive thinking"}],
            "thinking": {"type": "adaptive"},
        }
    )
    payload = _assert_anthropic_error_shape(response, expected_status=400)
    message = payload["error"].get("message", "")
    assert "does not support adaptive thinking" in message, payload


async def test_messages_rejects_invalid_thinking_type() -> None:
    response = await _post_messages_in_process(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use thinking"}],
            "thinking": {"type": "foo"},
        }
    )
    payload = _assert_anthropic_error_shape(response, expected_status=400)
    message = payload["error"].get("message", "")
    assert "thinking" in message, payload


async def test_streaming_event_order_and_shapes() -> None:
    if not await _require_provider_or_skip("streaming event order check"):
        return

    lines: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/messages",
                headers=ANTHROPIC_HEADERS,
                json={
                    "model": ANTHROPIC_STREAM_MODEL,
                    "max_tokens": 96,
                    "messages": [{"role": "user", "content": "Tell me a short joke."}],
                    "stream": True,
                },
            ) as response:
                if _is_upstream_unavailable(response):
                    print("- streaming event order check skipped (upstream provider unavailable)")
                    return
                assert response.status_code == 200, response.text
                _assert_request_id_headers(response)
                lines = [line async for line in response.aiter_lines()]
    except httpx.TimeoutException:
        print("- streaming event order check skipped (upstream provider timeout)")
        return

    events = _extract_stream_events(lines)
    assert events, "No streaming events received"

    event_names = [item["event"] for item in events]
    assert "message_start" in event_names, event_names
    assert "message_delta" in event_names, event_names
    assert "message_stop" in event_names, event_names

    start_index = event_names.index("message_start")
    delta_index = event_names.index("message_delta")
    stop_index = event_names.index("message_stop")
    assert start_index < delta_index < stop_index, event_names

    for item in events:
        if item["event"] == "message_delta":
            delta_payload = item["data"]
            assert delta_payload["type"] == "message_delta", delta_payload
            assert isinstance(delta_payload.get("usage"), dict), delta_payload
            assert "stop_reason" in delta_payload.get("delta", {}), delta_payload


async def test_non_stream_response_shape() -> None:
    if not await _require_provider_or_skip("non-stream response shape check"):
        return

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await _post_messages(
                client,
                {
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Say hi"}],
                },
            )
        except httpx.TimeoutException:
            print("- non-stream response shape check skipped (upstream provider timeout)")
            return

        if _is_upstream_unavailable(response):
            print("- non-stream response shape check skipped (upstream provider unavailable)")
            return

        assert response.status_code == 200, response.text
        _assert_request_id_headers(response)
        payload = response.json()
        assert payload.get("type") == "message", _pretty(payload)
        assert payload.get("role") == "assistant", _pretty(payload)
        assert isinstance(payload.get("content"), list), _pretty(payload)
        assert isinstance(payload.get("usage"), dict), _pretty(payload)
        assert payload.get("stop_reason") in {
            "end_turn",
            "max_tokens",
            "tool_use",
            "stop_sequence",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
            "error",
        }, _pretty(payload)


async def test_openai_chat_completion_endpoint_removed() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_openai_chat(
            client,
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 32,
            },
        )

    payload = _assert_openai_error_shape(response, expected_status=404)
    assert payload["error"].get("code") == "not_found", payload


async def test_openai_chat_stream_endpoint_removed() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_openai_chat(
            client,
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Tell me a short joke."}],
                "stream": True,
                "max_tokens": 64,
            },
        )

    payload = _assert_openai_error_shape(response, expected_status=404)
    assert payload["error"].get("code") == "not_found", payload


async def test_openai_responses_endpoint_removed() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_openai_responses(
            client,
            {
                "model": config.small_model,
                "input": "Say hello in one sentence",
                "max_output_tokens": 32,
            },
        )

    payload = _assert_openai_error_shape(response, expected_status=404)
    assert payload["error"].get("code") == "not_found", payload


async def test_openai_responses_stream_endpoint_removed() -> None:
    async with httpx.AsyncClient() as client:
        response = await _post_openai_responses(
            client,
            {
                "model": config.small_model,
                "input": "Say hello",
                "stream": True,
                "max_output_tokens": 32,
            },
        )

    payload = _assert_openai_error_shape(response, expected_status=404)
    assert payload["error"].get("code") == "not_found", payload


async def test_openai_model_supports_passthrough_prefixes() -> None:
    assert _is_supported_openai_model_id("gpt-5.2") is True
    assert _is_supported_openai_model_id("o3-mini") is True
    assert _is_supported_openai_model_id("deepseek-chat") is True
    assert _is_supported_openai_model_id("__definitely_unknown_model__") is False


async def test_openai_models_endpoints() -> None:
    async with httpx.AsyncClient() as client:
        list_response = await client.get(f"{BASE_URL}/v1/models", headers=OPENAI_HEADERS)
        assert list_response.status_code == 200, list_response.text
        _assert_request_id_headers(list_response)
        payload = list_response.json()
        assert payload.get("object") == "list", payload
        assert isinstance(payload.get("data"), list), payload
        assert payload["data"], payload

        first_model = payload["data"][0]["id"]
        get_response = await client.get(
            f"{BASE_URL}/v1/models/{first_model}", headers=OPENAI_HEADERS
        )
        assert get_response.status_code == 200, get_response.text
        _assert_request_id_headers(get_response)
        model_payload = get_response.json()
        assert model_payload.get("id") == first_model, model_payload


async def test_openai_models_missing_returns_openai_error() -> None:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/v1/models/not-a-real-model",
            headers=OPENAI_HEADERS,
        )

    payload = _assert_openai_error_shape(response, expected_status=404)
    assert payload["error"].get("code") == "model_not_found", payload


async def test_openai_models_auth_required() -> None:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/v1/models",
            headers={"content-type": "application/json"},
        )

    _assert_openai_error_shape(response, expected_status=401)


async def test_anthropic_uses_request_too_large_type_for_413_mapping() -> None:
    response = httpx.Response(
        status_code=413,
        json={
            "type": "error",
            "error": {"type": "request_too_large", "message": "x"},
            "request_id": "req_test",
        },
        headers={"request-id": "req_test", "x-request-id": "req_test"},
    )
    payload = _assert_anthropic_error_shape(response, expected_status=413)
    assert payload["error"]["type"] == "request_too_large", payload


async def test_request_converter_supports_image_url_sources() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/image.png",
                        },
                    },
                ],
            }
        ],
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    converted = convert_claude_to_openai(request_model, model_manager)
    input_items = converted.get("input")
    assert isinstance(input_items, list) and input_items, converted

    first_user = next(
        item
        for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user"
    )
    content = first_user.get("content")
    assert isinstance(content, list), first_user

    image_parts = [
        part for part in content if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    assert image_parts, first_user
    assert image_parts[0].get("image_url") == "https://example.com/image.png", first_user


def test_request_parser_accepts_cache_control_on_tool_blocks() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "system": [
            {
                "type": "text",
                "text": "System guidance",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "bash",
                        "input": {"command": "ls"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "AGENTS.md\nbackend\nfrontend\n",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
    }

    request_model = _build_claude_request_model(payload)
    assistant_block = request_model.messages[1].content[0]
    user_block = request_model.messages[2].content[0]

    assert getattr(assistant_block, "type", None) == "tool_use", request_model
    assert getattr(user_block, "type", None) == "tool_result", request_model
    assert getattr(assistant_block, "cache_control", None) is not None, request_model
    assert getattr(user_block, "cache_control", None) is not None, request_model


def test_request_converter_uses_output_text_for_assistant_history() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ],
    }

    request_model = _build_claude_request_model(payload)
    converted = convert_claude_to_openai(request_model, model_manager)
    input_items = converted.get("input")
    assert isinstance(input_items, list) and len(input_items) == 3, converted

    user_item = input_items[0]
    assistant_item = input_items[1]
    assert user_item.get("role") == "user", converted
    assert assistant_item.get("role") == "assistant", converted

    user_content = user_item.get("content")
    assistant_content = assistant_item.get("content")
    assert isinstance(user_content, list) and user_content, user_item
    assert isinstance(assistant_content, list) and assistant_content, assistant_item
    assert user_content[0].get("type") == "input_text", user_item
    assert assistant_content[0].get("type") == "output_text", assistant_item


async def test_request_converter_rejects_native_web_search_tool() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Get current weather"}],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2,
            }
        ],
        "tool_choice": {"type": "auto"},
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    try:
        convert_claude_to_openai(request_model, model_manager)
    except ValueError as error:
        assert "web_search" in str(error), error
    else:
        raise AssertionError("Expected web_search tool conversion to be rejected")


async def test_request_converter_maps_function_tools_to_responses_shape() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Use weather tool"}],
        "tools": [
            {
                "name": "weather",
                "description": "Get weather by city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    converted = convert_claude_to_openai(request_model, model_manager)
    tools = converted.get("tools")
    assert isinstance(tools, list) and tools, converted
    first_tool = tools[0]
    assert first_tool.get("type") == "function", first_tool
    assert first_tool.get("name") == "weather", first_tool
    assert first_tool.get("description") == "Get weather by city", first_tool
    assert first_tool.get("parameters", {}).get("type") == "object", first_tool
    assert "function" not in first_tool, first_tool


async def test_request_converter_maps_context_management() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "context_management": {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": {"type": "thinking_turns", "value": 2},
                }
            ]
        },
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    converted = convert_claude_to_openai(request_model, model_manager)
    assert converted.get("context_management") == [
        {"type": "compaction", "compact_threshold": 200000}
    ], converted


async def test_request_converter_rejects_top_k() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "top_k": 5,
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    try:
        convert_claude_to_openai(request_model, model_manager)
    except ValueError as error:
        assert "top_k" in str(error), error
    else:
        raise AssertionError("Expected top_k to be rejected in Responses-only mode")


async def test_request_converter_rejects_stop_sequences() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "stop_sequences": ["###"],
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    try:
        convert_claude_to_openai(request_model, model_manager)
    except ValueError as error:
        assert "stop_sequences" in str(error), error
    else:
        raise AssertionError("Expected stop_sequences to be rejected in Responses-only mode")


async def test_request_converter_rejects_inference_geo() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "inference_geo": {"type": "approximate", "country": "FI"},
    }

    request_model = parse_claude_messages_request(
        payload, allow_unknown_fields=config.anthropic_allow_unknown_fields
    )
    try:
        convert_claude_to_openai(request_model, model_manager)
    except ValueError as error:
        assert "inference_geo" in str(error), error
    else:
        raise AssertionError("Expected inference_geo to be rejected in Responses-only mode")


async def test_streaming_converter_handles_bad_request_before_first_chunk() -> None:
    request_payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    async def raising_stream() -> Any:
        if False:
            yield "data: ignored"
        raise HTTPException(status_code=400, detail="Error code: 400 - unsupported parameter")

    events = await _collect_sse_events(raising_stream(), request_payload)
    assert events, events
    assert events[0]["event"] == "message_start", events
    assert events[-1]["event"] == "error", events
    error_payload = events[-1]["data"].get("error", {})
    assert error_payload.get("type") == "api_error", events
    assert "unsupported parameter" in error_payload.get("message", ""), events


async def test_streaming_reasoning_and_multichoice_selection() -> None:
    request_payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Think briefly"}],
        "stream": True,
    }
    stream_lines = [
        'data: {"choices":[{"index":1,"delta":{"content":"skip-me"}}]}',
        'data: {"choices":[{"index":0,"delta":{"reasoning":"Plan"}}]}',
        'data: {"choices":[{"index":0,"delta":{"content":"Answer"}}]}',
        'data: {"choices":[{"index":0,"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]

    events = await _collect_sse_events(_iter_lines(stream_lines), request_payload)
    event_names = [event["event"] for event in events]
    assert "content_block_start" in event_names, event_names
    assert "message_delta" in event_names, event_names
    assert "message_stop" in event_names, event_names

    thinking_deltas = [
        event
        for event in events
        if event["event"] == "content_block_delta"
        and event["data"].get("delta", {}).get("type") == "thinking_delta"
    ]
    assert thinking_deltas, events
    assert any(
        event["data"]["delta"].get("thinking") == "Plan" for event in thinking_deltas
    ), thinking_deltas

    text_deltas = [
        event
        for event in events
        if event["event"] == "content_block_delta"
        and event["data"].get("delta", {}).get("type") == "text_delta"
    ]
    assert any(event["data"]["delta"].get("text") == "Answer" for event in text_deltas), text_deltas
    assert all(
        event["data"]["delta"].get("text") != "skip-me" for event in text_deltas
    ), text_deltas


async def test_streaming_tool_call_multidelta_interleaving() -> None:
    request_payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "call weather"}],
        "stream": True,
    }
    stream_lines = [
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","function":{"name":"weather"}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\""}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Helsinki"}}]}}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]}}]}',
        'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]

    events = await _collect_sse_events(_iter_lines(stream_lines), request_payload)

    start_events = [
        event
        for event in events
        if event["event"] == "content_block_start"
        and event["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(start_events) == 1, events
    content_block = start_events[0]["data"]["content_block"]
    assert content_block.get("id") == "call_weather", content_block
    assert content_block.get("name") == "weather", content_block

    input_deltas = [
        event["data"]["delta"].get("partial_json", "")
        for event in events
        if event["event"] == "content_block_delta"
        and event["data"].get("delta", {}).get("type") == "input_json_delta"
    ]
    assert input_deltas, events
    assert "".join(input_deltas) == '{"city":"Helsinki"}', input_deltas

    message_delta_events = [event for event in events if event["event"] == "message_delta"]
    assert message_delta_events, events
    assert (
        message_delta_events[-1]["data"].get("delta", {}).get("stop_reason") == "tool_use"
    ), message_delta_events[-1]


async def test_streaming_finish_reason_mapping_matrix() -> None:
    cases = [
        ("stop", None, "end_turn", None),
        ("stop", "###", "stop_sequence", "###"),
        ("stop_sequence", "###", "stop_sequence", "###"),
        ("length", None, "max_tokens", None),
        ("max_tokens", None, "max_tokens", None),
        ("tool_calls", None, "tool_use", None),
        ("function_call", None, "tool_use", None),
        ("pause_turn", None, "pause_turn", None),
        ("refusal", None, "refusal", None),
        ("content_filter", None, "refusal", None),
        ("model_context_window_exceeded", None, "model_context_window_exceeded", None),
    ]

    for finish_reason, stop_sequence, expected_reason, expected_sequence in cases:
        request_payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
        chunk_choice: Dict[str, Any] = {"index": 0, "finish_reason": finish_reason}
        if stop_sequence is not None:
            chunk_choice["stop_sequence"] = stop_sequence
        stream_lines = [
            f"data: {json.dumps({'choices': [chunk_choice]}, ensure_ascii=False)}",
            "data: [DONE]",
        ]
        events = await _collect_sse_events(_iter_lines(stream_lines), request_payload)
        message_delta = next(event for event in events if event["event"] == "message_delta")
        delta = message_delta["data"].get("delta", {})
        assert delta.get("stop_reason") == expected_reason, (finish_reason, events)
        assert delta.get("stop_sequence") == expected_sequence, (finish_reason, events)


async def test_tool_result_document_url_and_file_normalization() -> None:
    normalized = parse_tool_result_content(
        [
            {
                "type": "document",
                "title": "Reference",
                "source": {
                    "type": "url",
                    "url": "https://example.com/spec",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": "file_123",
                },
            },
        ]
    )
    assert "https://example.com/spec" in normalized, normalized
    assert "file_123" in normalized, normalized


def test_request_converter_store_flag_matches_state_mode() -> None:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request_model = _build_claude_request_model(payload)
    converted = convert_claude_to_openai(request_model, model_manager)
    expected_store = config.openai_responses_state_mode == "provider"
    assert converted.get("store") == expected_store, converted


def test_non_stream_converter_maps_responses_output_items() -> None:
    request_payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request_model = _build_claude_request_model(request_payload)
    openai_response = {
        "id": "resp_test_123",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need weather lookup"}],
            },
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Helsinki"}',
            },
            {
                "type": "message",
                "id": "msg_resp_1",
                "content": [{"type": "output_text", "text": "Here is the weather."}],
            },
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 9,
            "total_tokens": 21,
        },
    }

    claude_response = convert_openai_to_claude_response(openai_response, request_model)
    assert claude_response["id"] == "msg_resp_1", claude_response
    assert claude_response["type"] == "message", claude_response
    assert claude_response["role"] == "assistant", claude_response
    assert claude_response["stop_reason"] == "tool_use", claude_response
    assert claude_response["usage"]["input_tokens"] == 12, claude_response
    assert claude_response["usage"]["output_tokens"] == 9, claude_response

    blocks = claude_response["content"]
    assert isinstance(blocks, list) and blocks, claude_response
    assert any(block.get("type") == "thinking" for block in blocks), claude_response
    assert any(block.get("type") == "tool_use" for block in blocks), claude_response
    assert any(block.get("type") == "text" for block in blocks), claude_response


async def main() -> None:
    print("Running strict compatibility integration tests")
    await _resolve_runtime_models()
    print(
        f"Using Anthropic test models: default={ANTHROPIC_MODEL}, stream={ANTHROPIC_STREAM_MODEL}"
    )

    await test_missing_anthropic_version_header()
    print("- missing anthropic-version header check passed")

    test_openai_client_normalizes_root_base_url_to_v1()
    print("- openai client base-url normalization check passed")

    test_openai_client_preserves_explicit_versioned_base_url()
    print("- openai client explicit base-url passthrough check passed")

    test_openai_client_keeps_raw_base_url_for_azure_mode()
    print("- openai client azure-mode base-url passthrough check passed")

    await test_retries_without_metadata_on_unsupported_parameter_error()
    print("- metadata fallback (metadata) retry check passed")

    await test_retries_without_context_management_on_unsupported_parameter_error()
    print("- metadata fallback (context_management) retry check passed")

    await test_retries_without_extra_body_on_unsupported_parameter_error()
    print("- metadata fallback (extra_body) retry check passed")

    await test_retries_once_without_all_optional_fields_when_any_optional_field_is_rejected()
    print("- metadata fallback removes optional fields in single retry check passed")

    await test_does_not_retry_for_other_bad_request_errors()
    print("- metadata fallback does not retry unrelated bad requests check passed")

    await test_retries_without_context_management_on_retryable_server_error()
    print(
        "- metadata fallback retries on retryable server errors with context_management check passed"
    )

    await test_retries_without_proxy_metadata_context_management_on_retryable_server_error()
    print(
        "- metadata fallback retries on retryable server errors with proxy metadata context_management check passed"
    )

    await test_retries_without_extra_body_on_retryable_server_error()
    print("- metadata fallback retries on retryable server errors with extra_body check passed")

    test_retryable_server_status_includes_cloudflare_and_529()
    print("- retryable server status set includes cloudflare 52x and 529 check passed")

    test_normalize_error_status_code_maps_non_error_to_502()
    print("- status-code normalization for protocol errors check passed")

    test_protocol_error_classification_excludes_transport_failures()
    print("- protocol error classification excludes transport failures check passed")
    test_content_type_json_detection()
    print("- content-type JSON detection check passed")

    test_html_error_payload_signature_detection()
    print("- HTML error payload signature detection check passed")

    test_safe_payload_preview_normalizes_and_clips()
    print("- payload preview normalization/clipping check passed")

    test_build_upstream_protocol_error_uses_status_502_for_fake_200()
    print("- upstream protocol error mapping converts fake 200 to 502 check passed")

    test_build_upstream_protocol_error_keeps_upstream_error_status()
    print("- upstream protocol error mapping keeps valid upstream status check passed")

    await test_create_response_maps_protocol_validation_error_to_http_502()
    print("- non-stream protocol validation error maps to HTTP 502 check passed")

    await test_create_response_stream_maps_protocol_validation_error_to_http_502()
    print("- stream protocol validation error maps to HTTP 502 check passed")

    await test_create_response_maps_json_decode_error_to_http_502()
    print("- non-stream JSON decode protocol error maps to HTTP 502 check passed")

    await test_create_response_stream_maps_json_decode_error_to_http_502()
    print("- stream JSON decode protocol error maps to HTTP 502 check passed")

    await test_create_response_maps_bad_response_body_api_error_to_protocol_error()
    print("- non-stream bad_response_body API error maps to protocol error check passed")
    await test_retries_without_optional_fields_on_protocol_api_error()
    print(
        "- metadata fallback retries on protocol API error and strips optional fields check passed"
    )

    await test_retries_once_on_json_decode_protocol_error_without_optional_fields()
    print(
        "- metadata fallback retries once on JSON parse failure and strips optional fields check passed"
    )

    await test_does_not_retry_for_non_retryable_api_status_error()
    print("- metadata fallback does not retry non-retryable API status errors check passed")

    await test_lightweight_client_cancellation_for_non_stream_response()
    print("- lightweight non-stream cancellation check passed")

    await test_unknown_fields_rejected_with_anthropic_error_shape()
    print("- unknown fields rejection shape check passed")

    await test_invalid_role_content_rejected_with_anthropic_error_shape()
    print("- role/content validation check passed")

    await test_anthropic_large_request_returns_413()
    print("- anthropic request-size guard check passed")

    await test_token_count_endpoint_header_and_response_shape()
    print("- count_tokens shape check passed")

    await test_token_count_includes_tools_and_thinking()
    print("- count_tokens tools/thinking weighting check passed")

    await test_token_count_rejects_web_search_conflicting_domain_filters()
    print("- count_tokens web_search domain filter validation check passed")

    await test_messages_accepts_adaptive_thinking_for_claude_sonnet_4_6()
    print("- messages adaptive thinking support (claude-sonnet-4-6) check passed")

    await test_messages_maps_low_effort_to_openai_medium()
    print("- messages adaptive low effort maps to medium check passed")

    await test_messages_accepts_adaptive_thinking_for_claude_opus_4_6()
    print("- messages adaptive thinking support (claude-opus-4-6) check passed")

    await test_messages_omitted_temperature_is_not_forwarded()
    print("- messages omitted temperature passthrough check passed")

    await test_messages_explicit_temperature_is_forwarded()
    print("- messages explicit temperature passthrough (thinking.disabled) check passed")

    await test_messages_explicit_temperature_requires_thinking_disabled()
    print("- messages explicit temperature requires thinking.disabled check passed")

    await test_messages_explicit_temperature_scales_to_x2_when_enabled()
    print("- messages explicit temperature x2 scaling check passed")

    await test_messages_drops_sampling_fields_when_reasoning_conflicts_by_default()
    print("- messages default drop_sampling conflict handling check passed")

    await test_messages_force_reasoning_none_when_sampling_conflicts()
    print("- messages force_reasoning_none conflict handling check passed")

    await test_messages_strict_error_when_sampling_conflicts()
    print("- messages strict_error conflict handling check passed")

    await test_messages_non_gpt5_model_applies_sampling_compatibility()
    print("- messages non-gpt5 sampling compatibility check passed")

    await test_messages_maps_max_effort_to_openai_xhigh()
    print("- messages adaptive max effort maps to xhigh check passed")

    await test_messages_enabled_thinking_with_budget_tokens_still_works_for_sonnet_4_5()
    print("- messages Sonnet 4.5 compatibility (enabled + budget_tokens) check passed")

    await test_messages_enabled_thinking_budget_tokens_maps_to_xhigh_without_output_config()
    print("- messages budget_tokens mapping to xhigh check passed")

    await test_messages_output_config_without_effort_falls_back_to_thinking_budget()
    print("- messages empty output_config falls back to budget mapping check passed")

    await test_messages_enabled_thinking_without_budget_turns_reasoning_off()
    print("- messages enabled thinking without budget keeps reasoning off check passed")

    await test_messages_rejects_adaptive_thinking_for_older_models()
    print("- messages adaptive thinking model compatibility rejection check passed")

    await test_messages_rejects_invalid_thinking_type()
    print("- messages invalid thinking.type validation check passed")

    await test_request_converter_supports_image_url_sources()
    print("- request converter image url support check passed")
    test_request_converter_uses_output_text_for_assistant_history()
    print("- request converter assistant history text-type check passed")

    await test_request_converter_rejects_native_web_search_tool()
    print("- request converter native web_search rejection check passed")

    await test_request_converter_maps_function_tools_to_responses_shape()
    print("- request converter function-tools shape check passed")

    await test_request_converter_maps_context_management()
    print("- request converter context_management mapping check passed")

    await test_request_converter_rejects_top_k()
    print("- request converter top_k rejection check passed")
    await test_request_converter_rejects_stop_sequences()
    print("- request converter stop_sequences rejection check passed")

    await test_request_converter_rejects_inference_geo()
    print("- request converter inference_geo rejection check passed")

    await test_tool_result_document_url_and_file_normalization()
    print("- tool_result document url/file normalization check passed")

    test_request_converter_store_flag_matches_state_mode()
    print("- request converter store flag state-mode check passed")

    test_non_stream_converter_maps_responses_output_items()
    print("- non-stream converter Responses output mapping check passed")

    await test_streaming_converter_handles_bad_request_before_first_chunk()
    print("- streaming converter pre-first-chunk bad-request handling check passed")

    await test_streaming_reasoning_and_multichoice_selection()
    print("- streaming reasoning and multi-choice selection check passed")

    await test_streaming_tool_call_multidelta_interleaving()
    print("- streaming tool-call interleaving check passed")

    await test_streaming_finish_reason_mapping_matrix()
    print("- streaming finish-reason mapping matrix check passed")

    await test_non_stream_response_shape()
    print("- non-stream response shape check passed")

    await test_streaming_event_order_and_shapes()
    print("- streaming event order check passed")

    await test_openai_chat_completion_endpoint_removed()
    print("- openai chat endpoint removed check passed")

    await test_openai_chat_stream_endpoint_removed()
    print("- openai chat stream endpoint removed check passed")

    await test_openai_responses_endpoint_removed()
    print("- openai responses endpoint removed check passed")

    await test_openai_responses_stream_endpoint_removed()
    print("- openai responses stream endpoint removed check passed")

    await test_openai_model_supports_passthrough_prefixes()
    print("- openai model support-prefix check passed")

    await test_openai_models_endpoints()
    print("- openai models endpoints check passed")

    await test_openai_models_missing_returns_openai_error()
    print("- openai model-not-found error shape check passed")

    await test_openai_models_auth_required()
    print("- openai models auth required check passed")

    await test_anthropic_uses_request_too_large_type_for_413_mapping()
    print("- anthropic 413 error type mapping check passed")

    print("All strict compatibility tests passed")


if __name__ == "__main__":
    asyncio.run(main())
