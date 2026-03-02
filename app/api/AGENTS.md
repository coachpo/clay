# API KNOWLEDGE BASE

## OVERVIEW

`endpoints.py` hosts Anthropic-compatible generation routes, OpenAI-compatible model discovery routes, and validation/cancellation glue.

## WHERE TO LOOK

- Route entrypoints: `create_message`, `count_tokens`, `list_models_openai`, `get_model_openai`.
- Validation gates: `validate_api_contract` and `validate_openai_api_contract`.
- Version policy: `_resolve_requested_anthropic_version`.
- Disconnect handling: `_create_response_with_disconnect_cancellation` and `_wait_for_http_disconnect`.
- Error shaping: `_anthropic_error_response`, `_openai_error_response`, status-type mappers.

## LOCAL CONTRACTS

- Anthropic request size guard enforces 32 MB via `content-length` before body parsing.
- Anthropic version acceptance is driven by config (`ANTHROPIC_SUPPORTED_VERSIONS`, default `2023-06-01`); missing header behavior is flag-controlled.
- Non-JSON content types for Anthropic routes are rejected.
- Response headers must include both `request-id` and `x-request-id` on success and errors.
- Streaming `/v1/messages` delegates SSE conversion to `convert_openai_streaming_to_claude_with_cancellation`.
- Non-stream `/v1/messages` path must keep disconnect-aware cancellation behavior.
- Removed generation surfaces `POST /v1/chat/completions` and `POST /v1/responses` return 404 by design.
- OpenAI-compatible auth guard applies to `/v1/models` and `/v1/models/{model_id}`.

## CONVENTIONS

- Keep Anthropic and OpenAI error payload shapes distinct and status-appropriate.
- Generate one `request_id` per request and thread it through transport/conversion/cancellation.
- Prefer small targeted fixes in this file; many routes share helper behavior.

## ANTI-PATTERNS

- Do not bypass `Depends(validate_api_contract)` or `Depends(validate_openai_api_contract)` on protected routes.
- Do not change SSE event ordering rules here without synchronized conversion/test updates.
- Do not remove request-id headers from error paths.
- Do not re-enable removed OpenAI generation endpoints without full contract and converter updates.

## VERIFICATION

```bash
python tests/test_main.py
```
