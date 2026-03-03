# API KNOWLEDGE BASE

## OVERVIEW
`endpoints.py` owns request contract enforcement, Anthropic generation lifecycle, OpenAI-compatible model discovery, and disconnect-aware cancellation glue.

## WHERE TO LOOK
- Main Anthropic path: `create_message()` (`POST /v1/messages`).
- Token counting: `count_tokens()` (`POST /v1/messages/count_tokens`) + `_estimate_*` helpers.
- OpenAI model discovery: `list_models_openai()` and `get_model_openai()` (`GET /v1/models*`).
- Contract validators: `validate_api_contract()` and `validate_openai_api_contract()`.
- Version handling: `_resolve_requested_anthropic_version()` + fallback helpers.
- Disconnect cancellation: `_create_response_with_disconnect_cancellation()` + `_wait_for_http_disconnect()`.
- Error payload shaping: `_anthropic_error_response()`, `_openai_error_response()`, status-to-type mappers.

## LOCAL CONTRACTS
- Anthropic request-size guard enforces 32 MB by `content-length` before body parsing (`MAX_ANTHROPIC_REQUEST_BYTES`).
- Anthropic header/version policy is config-driven (`ANTHROPIC_SUPPORTED_VERSIONS`, missing/fallback flags, default `2023-06-01`).
- Anthropic routes reject non-JSON `content-type` values.
- `request_id` is generated per request and returned in both `request-id` and `x-request-id` headers.
- Streaming `/v1/messages` uses `convert_openai_streaming_to_claude_with_cancellation()` and must preserve Claude SSE order.
- Non-stream `/v1/messages` path must retain disconnect-aware cancellation behavior (`499` on client disconnect).
- Adaptive thinking is restricted to model prefixes in `ADAPTIVE_THINKING_MODEL_PREFIXES` (`claude-opus-4-6`, `claude-sonnet-4-6`).
- OpenAI-compatible generation endpoints `/v1/chat/completions` and `/v1/responses` intentionally return 404.
- OpenAI-compatible auth gate protects `/v1/models` and `/v1/models/{model_id}`.
- Optional Anthropic compatibility metadata may be attached into provider payload `extra_body.proxy_metadata`.

## CONVENTIONS
- Keep Anthropic and OpenAI error envelopes distinct and status-mapped (`invalid_request_error`, `api_error`, etc.).
- Keep `request_id` threading consistent across API, conversion, and client cancellation maps.
- Keep bugfixes minimal in this file; helpers are shared by multiple routes and tests assert exact shapes.

## ANTI-PATTERNS
- Do not bypass `Depends(validate_api_contract)` or `Depends(validate_openai_api_contract)` on protected routes.
- Do not change SSE ordering/event names without synchronized converter + integration test updates.
- Do not remove request-id header parity from success or error responses.
- Do not silently re-enable removed OpenAI generation surfaces without full contract/converter/test updates.

## VERIFICATION
```bash
python tests/test_main.py
```
