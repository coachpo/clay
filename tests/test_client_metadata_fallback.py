from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import httpx
import pytest
from openai._exceptions import BadRequestError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.client import OpenAIClient


def _bad_request_error(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://example.com/v1/responses")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(
        message=message,
        response=response,
        body={"error": {"message": message}},
    )


@pytest.mark.asyncio
async def test_retries_without_metadata_on_unsupported_parameter_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: list[Dict[str, Any]] = []
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


@pytest.mark.asyncio
async def test_retries_without_context_management_on_unsupported_parameter_error() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: list[Dict[str, Any]] = []
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


@pytest.mark.asyncio
async def test_retries_once_without_all_optional_fields_when_any_optional_field_is_rejected() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: list[Dict[str, Any]] = []
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


@pytest.mark.asyncio
async def test_does_not_retry_for_other_bad_request_errors() -> None:
    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")
    calls: list[Dict[str, Any]] = []
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

    with pytest.raises(BadRequestError):
        await client._create_with_metadata_fallback(request)

    assert len(calls) == 1
