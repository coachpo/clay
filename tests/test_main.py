import json
import os
from typing import Any

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "openai-test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.test/v1")
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

FIRST_SYSTEM_PROMPT = (
    "You are a senior platform engineer. Give clear, practical, structured answers. "
    "Do not ask clarifying questions."
)

FIRST_USER_PROMPT = (
    "I am migrating a FastAPI proxy from chat completions to the OpenAI Responses API. "
    "Constraints: 2-day deadline, one engineer, must preserve Anthropic-compatible "
    "/v1/messages behavior, support streaming, and minimize breaking changes. Give: "
    "1) a phased plan, 2) biggest technical risks, 3) tests to run before release."
)

FIRST_ASSISTANT_TEXT = """Use an adapter-first migration: keep `/v1/messages` stable, swap the upstream core behind it, and defer any schema cleanup that is not required for correctness.

**Phased Plan**
- Phase 0 - Freeze the contract: inventory current `/v1/messages` request/response behavior, streaming event shapes, error mapping, tool-use semantics, stop reasons, token usage fields, and any quirks clients depend on; write this down before changing code.
- Phase 1 - Add a Responses adapter layer: create a narrow internal interface like `generate(request) -> normalized result|stream`, with one implementation for legacy chat completions and one for Responses; keep FastAPI handlers calling only the normalized layer.
- Phase 2 - Build request translation: map Anthropic-style messages into Responses API input items, including system prompts, user/assistant turns, tool definitions, tool results, max tokens, temperature, stop sequences, and metadata; preserve unsupported fields by ignoring or logging rather than failing unless they affect correctness.
- Phase 3 - Build response translation: convert Responses outputs back into Anthropic-compatible `/v1/messages` JSON, including `content` blocks, `stop_reason`, `usage`, tool-use blocks, and stable IDs; do the same for streaming SSE events so clients see the same event contract they already parse.
- Phase 4 - Ship streaming parity: implement Responses streaming consumption and re-emit Anthropic-compatible SSE events in the same order and granularity your clients expect; optimize for correctness over cleverness, including clean final events and disconnect handling.
- Phase 5 - Dual-run and gate rollout: keep both upstream paths behind a feature flag or env toggle; for a subset of requests, run Responses in shadow mode, compare normalized outputs/logs, and only switch default traffic once major mismatches are resolved.
- Phase 6 - Harden and cut scope: explicitly postpone nonessential changes like broad refactors, new API surface, or perfect field parity for obscure options; document known differences if any remain.

**Biggest Technical Risks**
- Streaming semantic mismatch: Responses streaming events do not line up 1:1 with Anthropic `/v1/messages` SSE expectations, so event ordering, partial text chunks, tool call chunks, and terminal events are the highest breakage risk.
- Message/content model mismatch: Anthropic content blocks and tool-use/tool-result structures differ from Responses input/output items; multi-part content, mixed text+tool turns, and assistant-prefilled content can be lossy if not normalized carefully.
- Stop reason and finish-state drift: mapping `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, refusals, and incomplete responses can easily produce subtle client regressions even when text output looks fine.
- Tool calling differences: tool schema shape, argument serialization, call IDs, and whether the model emits calls incrementally during streaming may not match current client assumptions.
- Usage/accounting differences: token counts and subfields may differ or arrive later than before; if downstream systems bill, rate-limit, or log based on current usage fields, this can cause operational issues.
- Error mapping regressions: upstream Responses errors, timeouts, validation failures, and content policy blocks need translation into the same HTTP codes and response bodies clients already expect.
- Stateful edge cases: long conversations, truncated context, retries, idempotency, and client disconnects during streaming can expose hidden coupling in the current proxy.

**Practical Implementation Shape**
- Add `AnthropicRequest -> NormalizedRequest -> Responses API` translation and `Responses API -> NormalizedResponse -> AnthropicResponse` translation so your web layer stays unchanged.
- Normalize around your own internal content model for `text`, `tool_call`, `tool_result`, `usage`, and `finish_reason`; this reduces ad hoc mapping bugs.
- Keep the old chat-completions implementation alive for fallback during rollout; with a 2-day deadline, reversibility matters more than elegance.
- Log side-by-side diffs for normalized outputs, not raw provider payloads; this makes mismatches actionable.
- Treat unsupported fields as no-ops unless they are contract-critical; failing closed on everything will slow the migration and break compatibility unnecessarily.

**Tests Before Release**
- Contract tests for `/v1/messages`: verify request validation, response shape, headers, status codes, and exact SSE event names/ordering against the current proxy behavior.
- Golden tests for representative conversations: plain text reply, multi-turn chat, system prompt, stop sequence, max token cutoff, tool call, tool result follow-up, refusal/safety case, and empty/minimal responses.
- Streaming tests: confirm chunked text assembly, tool-call streaming if supported, final message event, usage emission, heartbeat behavior if any, and client disconnect cleanup.
- Translation tests: unit-test mappings from Anthropic request fields to Responses payloads and back, including mixed content arrays and edge values like null/omitted fields.
- Error-path tests: upstream 4xx/5xx, timeout, malformed tool arguments, invalid model, context length overflow, and rate limiting; verify your proxy preserves expected HTTP codes and body schema.
- Regression tests for finish reasons: ensure `end_turn`, `max_tokens`, `stop_sequence`, and `tool_use` map consistently in both non-streaming and streaming modes.
- Load/smoke tests: run concurrent streaming and non-streaming requests through the proxy to catch connection leaks, backpressure issues, and latency spikes.
- Shadow comparison tests: replay a small corpus of real requests through both old and new paths, compare normalized outputs, and manually inspect the highest-diff cases.

**Recommended 2-Day Breakdown**
- Day 1 AM: document current contract and add normalized adapter interface.
- Day 1 PM: implement non-streaming request/response translation and basic parity tests.
- Day 2 AM: implement streaming translation, disconnect/error handling, and golden SSE tests.
- Day 2 PM: shadow run, fix biggest mismatches, add feature flag fallback, and release with rollback path.

**Release Bar**
- `/v1/messages` clients do not need code changes.
- Streaming works for normal text responses and tool-use flows.
- Known differences are documented and judged non-breaking.
- Fallback to legacy upstream path is one config change away.

If you want, I can turn this into a concrete FastAPI migration checklist with handler shapes, adapter interfaces, and SSE event mapping examples."""

SECOND_SYSTEM_PROMPT = "You are a senior platform engineer. Give concise, actionable answers."

SECOND_USER_PROMPT = """New facts: the upstream rejects metadata on /responses, stop_sequences are unsupported, custom OPENAI_BASE_URL support is required, and we must remove chat-completions usage entirely with no backward compatibility. Rewrite this as a shorter ordered execution plan with rollback points and explicit release criteria."""

SECOND_ASSISTANT_TEXT = """1. Lock the target contract for `/v1/messages`.
- Preserve Anthropic-compatible request/response and streaming behavior at the proxy boundary, but explicitly drop any internal backward compatibility for chat-completions.
- Document required field mappings and deltas now: omit `metadata` on upstream `/responses`, treat `stop_sequences` as proxy-local unsupported behavior, and require configurable `OPENAI_BASE_URL`.
- Rollback point: none in code yet; this is the go/no-go spec.
- Release criteria: written mapping for normal responses, streaming events, tool use, errors, usage, and unsupported fields.

2. Replace the upstream client with a single Responses-only adapter.
- Remove chat-completions code paths from the request pipeline and route all generation through one internal adapter built on `/responses`.
- Add `OPENAI_BASE_URL` support in the client configuration and verify it applies to both streaming and non-streaming calls.
- Ensure the adapter strips `metadata` before upstream submission.
- Rollback point: keep this in one isolated module so you can revert the migration commit cleanly if needed.
- Release criteria: no runtime path references chat-completions; base URL override works; upstream accepts all supported requests.

3. Implement request translation from Anthropic messages to Responses input.
- Map system, user, assistant, tool definitions, and tool results into a normalized internal model and then into `/responses`.
- Handle `stop_sequences` explicitly: ignore with logging or reject with a clear 4xx, but do it consistently and document it.
- Preserve minimal breaking changes by keeping the external `/v1/messages` schema unchanged wherever possible.
- Rollback point: feature branch or revert of translator commit if parity is not reached.
- Release criteria: representative non-streaming requests translate successfully, and unsupported fields fail or no-op predictably.

4. Implement response translation back to Anthropic-compatible `/v1/messages`.
- Convert `/responses` output into the existing Anthropic-style JSON shape, including content blocks, stop reason, usage, and tool-use content.
- Normalize finish reasons carefully so clients see stable values even if upstream semantics differ.
- Rollback point: revert response-mapping commit if output contract breaks clients.
- Release criteria: golden responses match current `/v1/messages` expectations for text, tool use, max token cutoff, and errors.

5. Implement streaming parity on top of Responses streaming.
- Consume `/responses` stream events and re-emit Anthropic-compatible SSE events in the order and structure clients already expect.
- Prioritize correct final events, tool-call chunks if applicable, usage emission, and client disconnect cleanup.
- Rollback point: if streaming is not stable, do not release; there is no legacy fallback.
- Release criteria: existing streaming clients work without code changes for text and tool-use flows.

6. Remove chat-completions usage entirely.
- Delete dead code, old config, old client wrappers, and any compatibility shims that keep chat-completions alive.
- Search the repo for chat-completions references and remove them, including tests and docs that imply fallback behavior.
- Rollback point: revert the removal commit set if late issues appear before release.
- Release criteria: repo has no active chat-completions usage; tests and docs reflect Responses-only behavior.

7. Run focused release validation and ship behind a fast revert path.
- Test non-streaming, streaming, tool calling, error mapping, unsupported `stop_sequences`, omitted `metadata`, and custom `OPENAI_BASE_URL`.
- Do a short staging soak with real client traffic patterns if possible.
- Rollback point: revert the migration commits as a unit if staging or production shows contract breakage.
- Release criteria:
  - `/v1/messages` remains Anthropic-compatible for supported behavior.
  - All upstream calls use `/responses` only.
  - `metadata` is never sent upstream.
  - `stop_sequences` behavior is consistent and documented.
  - `OPENAI_BASE_URL` works in streaming and non-streaming modes.
  - Streaming passes contract tests and works with existing clients.
  - No known P0/P1 mismatches in tool use, finish reasons, error shape, or connection handling.

Recommended pre-release test set:
1. Non-streaming golden tests: simple text, multi-turn, system prompt, tool call, tool result, max tokens, refusal/error.
2. Streaming contract tests: event names, order, final event, partial text assembly, tool-use stream behavior, disconnect handling.
3. Config tests: custom `OPENAI_BASE_URL` for both request modes.
4. Negative tests: `metadata` stripped, `stop_sequences` ignored or rejected exactly as documented.
5. Repo check: grep confirms no remaining chat-completions runtime usage.

If you want, I can turn this into a 2-day task breakdown with owner-sized tickets and exact test names."""


def fail_acompletion(**_: Any) -> None:
    raise AssertionError("acompletion() should not be used")


class StrictDumpEvent:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str = "python", warnings: bool = True) -> dict[str, Any]:
        assert mode == "json"
        assert warnings is False
        return self.payload


def chunk_text(text: str, *, size: int = 97) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def parse_sse_events(lines: list[str]) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current_event = ""

    for line in lines:
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
            continue

        if not line.startswith("data: "):
            continue

        payload = json.loads(line[len("data: ") :])
        if isinstance(payload, dict):
            events.append((current_event, payload))

    return events


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "2.0.0"


def test_messages_route_uses_responses_api(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": "resp_test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "test"}],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        }

    monkeypatch.setattr(endpoints.litellm, "acompletion", fail_acompletion)
    monkeypatch.setattr(endpoints.litellm, "aresponses", fake_aresponses)

    response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "system": "Be concise.",
            "metadata": {"client_trace_id": "trace-123"},
            "messages": [{"role": "user", "content": "Say test"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["request-id"] == response.headers["x-request-id"]
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["timeout"] == config.request_timeout
    assert captured["api_key"] == "openai-test-key"
    assert captured["api_base"] == "https://example.test/v1"
    assert captured["max_output_tokens"] == 16
    assert captured["instructions"] == "Be concise."
    assert "metadata" not in captured
    assert "messages" not in captured
    assert captured["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say test"}],
        }
    ]
    assert response.json() == {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "test"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
    }


def test_messages_route_streams_responses_events(monkeypatch: Any) -> None:
    async def fake_stream() -> Any:
        yield {
            "type": "response.created",
            "response": {"id": "resp_stream", "usage": {"input_tokens": 3, "output_tokens": 0}},
        }
        yield {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_stream",
            "delta": "Hel",
        }
        yield {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_stream",
            "delta": "lo",
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_stream",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        }

    async def fake_aresponses(**_: Any) -> Any:
        return fake_stream()

    monkeypatch.setattr(endpoints.litellm, "acompletion", fail_acompletion)
    monkeypatch.setattr(endpoints.litellm, "aresponses", fake_aresponses)

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


def test_tool_call_response_is_converted_from_responses_output(monkeypatch: Any) -> None:
    async def fake_aresponses(**_: Any) -> dict[str, Any]:
        return {
            "id": "resp_tool",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "weather",
                    "arguments": '{"city":"Helsinki"}',
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        }

    monkeypatch.setattr(endpoints.litellm, "acompletion", fail_acompletion)
    monkeypatch.setattr(endpoints.litellm, "aresponses", fake_aresponses)

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


def test_stop_sequences_are_rejected_on_responses_path(monkeypatch: Any) -> None:
    monkeypatch.setattr(endpoints.litellm, "acompletion", fail_acompletion)

    async def fake_aresponses(**_: Any) -> dict[str, Any]:
        raise AssertionError("aresponses() should not be called for invalid requests")

    monkeypatch.setattr(endpoints.litellm, "aresponses", fake_aresponses)

    response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 16,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    )

    assert response.status_code == 400
    assert "stop_sequences" in response.json()["error"]["message"]


def test_exact_two_turn_live_transcript_regression(monkeypatch: Any) -> None:
    captured_calls: list[dict[str, Any]] = []

    async def fake_stream() -> Any:
        yield StrictDumpEvent(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_turn_2",
                    "usage": {"input_tokens": 2940, "output_tokens": 0, "total_tokens": 2940},
                },
            }
        )

        for text_chunk in chunk_text(SECOND_ASSISTANT_TEXT):
            yield StrictDumpEvent(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_turn_2",
                    "delta": text_chunk,
                }
            )

        yield StrictDumpEvent(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_turn_2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_turn_2",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": SECOND_ASSISTANT_TEXT}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 2940,
                        "output_tokens": 1102,
                        "total_tokens": 4042,
                    },
                },
            }
        )

    async def fake_aresponses(**kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        if kwargs.get("stream"):
            return fake_stream()

        return {
            "id": "resp_turn_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_turn_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": FIRST_ASSISTANT_TEXT}],
                }
            ],
            "usage": {
                "input_tokens": 1634,
                "output_tokens": 1306,
                "total_tokens": 2940,
            },
        }

    monkeypatch.setattr(endpoints.litellm, "acompletion", fail_acompletion)
    monkeypatch.setattr(endpoints.litellm, "aresponses", fake_aresponses)

    first_response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": FIRST_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": FIRST_USER_PROMPT}],
        },
    )

    assert first_response.status_code == 200
    assert first_response.json() == {
        "id": "msg_turn_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": FIRST_ASSISTANT_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1634,
            "output_tokens": 1306,
            "total_tokens": 2940,
        },
    }

    first_assistant_text = first_response.json()["content"][0]["text"]

    with client.stream(
        "POST",
        "/v1/messages",
        headers=HEADERS,
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": SECOND_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": FIRST_USER_PROMPT},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": first_assistant_text}],
                },
                {"role": "user", "content": SECOND_USER_PROMPT},
            ],
            "stream": True,
        },
    ) as second_response:
        second_lines = [line for line in second_response.iter_lines() if line]

    assert second_response.status_code == 200

    second_events = parse_sse_events(second_lines)
    second_event_names = [event_name for event_name, _ in second_events]
    streamed_text = "".join(
        payload["delta"]["text"]
        for event_name, payload in second_events
        if event_name == "content_block_delta"
        and payload.get("delta", {}).get("type") == "text_delta"
    )

    assert streamed_text == SECOND_ASSISTANT_TEXT
    assert second_event_names[0] == "message_start"
    assert second_event_names[1] == "ping"
    assert "content_block_start" in second_event_names
    assert "content_block_delta" in second_event_names
    assert "content_block_stop" in second_event_names
    assert second_event_names[-2] == "message_delta"
    assert second_event_names[-1] == "message_stop"
    assert any(
        payload.get("delta", {}).get("stop_reason") == "end_turn"
        for event_name, payload in second_events
        if event_name == "message_delta"
    )

    assert len(captured_calls) == 2
    assert captured_calls[0]["model"] == config.middle_model
    assert captured_calls[1]["model"] == config.middle_model
    assert captured_calls[0]["api_base"] == config.openai_base_url
    assert captured_calls[1]["api_base"] == config.openai_base_url
    assert captured_calls[0]["instructions"] == FIRST_SYSTEM_PROMPT
    assert captured_calls[1]["instructions"] == SECOND_SYSTEM_PROMPT
    assert captured_calls[0]["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": FIRST_USER_PROMPT}],
        }
    ]
    assert captured_calls[1]["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": FIRST_USER_PROMPT}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": FIRST_ASSISTANT_TEXT}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": SECOND_USER_PROMPT}],
        },
    ]
    assert captured_calls[0]["stream"] is False
    assert captured_calls[1]["stream"] is True
    assert "metadata" not in captured_calls[0]
    assert "metadata" not in captured_calls[1]
