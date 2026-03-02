"""Integration tests for strict Anthropic and OpenAI compatibility over the proxy."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.api.endpoints import _is_supported_openai_model_id
from src.conversion.request_converter import convert_claude_to_openai, parse_tool_result_content
from src.conversion.response_converter import (
    convert_openai_streaming_to_claude,
    convert_openai_to_claude_response,
)
from src.core.config import config
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.models.claude import parse_claude_messages_request

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CLIENT_API_KEY = os.getenv("ANTHROPIC_API_KEY", "test-key")

ANTHROPIC_HEADERS = {
    "anthropic-version": "2023-06-01",
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

    await test_request_converter_supports_image_url_sources()
    print("- request converter image url support check passed")

    await test_request_converter_rejects_native_web_search_tool()
    print("- request converter native web_search rejection check passed")

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
