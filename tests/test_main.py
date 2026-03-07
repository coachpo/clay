import os
from typing import Any

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "openai-test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "client-test-key")
os.environ.setdefault("BIG_MODEL", "gpt-4.1")
os.environ.setdefault("MIDDLE_MODEL", "gpt-4.1-mini")
os.environ.setdefault("SMALL_MODEL", "gpt-4.1-nano")

from app.api import endpoints
from app.core.config import config
from app.main import app

client = TestClient(app)

HEADERS = {
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "x-api-key": "client-test-key",
}


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "2.0.0"


def test_messages_route_returns_anthropic_shape(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": "chatcmpl_test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": "test",
                        "role": "assistant",
                    },
                }
            ],
            "usage": {"completion_tokens": 1, "prompt_tokens": 4},
        }

    monkeypatch.setattr(endpoints.litellm, "acompletion", fake_acompletion)

    response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say test"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["request-id"] == response.headers["x-request-id"]
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["timeout"] == config.request_timeout
    assert captured["api_key"] == "openai-test-key"
    assert response.json() == {
        "id": "chatcmpl_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "test"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }


def test_messages_route_streams_anthropic_events(monkeypatch: Any) -> None:
    async def fake_stream() -> Any:
        yield {
            "choices": [{"index": 0, "delta": {"content": "Hel"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        yield {
            "choices": [{"index": 0, "delta": {"content": "lo"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        yield {"choices": [{"index": 0, "finish_reason": "stop"}]}

    async def fake_acompletion(**_: Any) -> Any:
        return fake_stream()

    monkeypatch.setattr(endpoints.litellm, "acompletion", fake_acompletion)

    with client.stream(
        "POST",
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert "event: message_start" in lines
    assert "event: ping" in lines
    assert "event: content_block_start" in lines
    assert "event: content_block_delta" in lines
    assert "event: content_block_stop" in lines
    assert "event: message_delta" in lines
    assert "event: message_stop" in lines
    assert any('"text": "Hel"' in line for line in lines)
    assert any('"text": "lo"' in line for line in lines)


def test_invalid_api_key_returns_anthropic_error() -> None:
    response = client.post(
        "/v1/messages",
        headers={**HEADERS, "x-api-key": "wrong-key"},
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    )

    assert response.status_code == 401
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "authentication_error"


def test_missing_anthropic_version_returns_400() -> None:
    headers = dict(HEADERS)
    headers.pop("anthropic-version")

    response = client.post(
        "/v1/messages",
        headers=headers,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Missing required header: anthropic-version"


def test_non_object_body_returns_400() -> None:
    response = client.post("/v1/messages", headers=HEADERS, json=[{"role": "user"}])

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid JSON body: expected an object."


def test_tool_call_response_is_converted(monkeypatch: Any) -> None:
    async def fake_acompletion(**_: Any) -> dict[str, Any]:
        return {
            "id": "chatcmpl_tool",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Helsinki"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"completion_tokens": 3, "prompt_tokens": 4},
        }

    monkeypatch.setattr(endpoints.litellm, "acompletion", fake_acompletion)

    response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Weather?"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "call_weather",
            "name": "weather",
            "input": {"city": "Helsinki"},
        }
    ]
    assert response.json()["stop_reason"] == "tool_use"
