# API KNOWLEDGE BASE

## OVERVIEW
`endpoints.py` hosts Anthropic-compatible and OpenAI-compatible HTTP surfaces plus validation/cancellation glue.

## WHERE TO LOOK
- Route entrypoints: `create_message`, `count_tokens`, `create_chat_completion_openai`, `create_responses_openai`.
- Validation gates: `validate_api_contract` and `validate_openai_api_contract`.
- Disconnect handling: `_create_chat_completion_with_disconnect_cancellation` and `_wait_for_http_disconnect`.
- Error shaping: `_anthropic_error_response`, `_openai_error_response`, status-type mappers.

## LOCAL CONTRACTS
- Anthropic requests require `anthropic-version: 2023-06-01`; non-JSON content types are rejected.
- Anthropic request size guard enforces 32 MB via `content-length` before body processing.
- Response headers must include both `request-id` and `x-request-id` on success and errors.
- Streaming `/v1/messages` delegates SSE conversion to `convert_openai_streaming_to_claude_with_cancellation`.
- Non-stream `/v1/messages` path must keep disconnect-aware cancellation behavior.
- `/v1/responses` is implemented as an adapter over chat completions; keep mapping logic aligned both ways.

## CONVENTIONS
- Keep Anthropic and OpenAI error payload shapes distinct and status-appropriate.
- Use `request_id` generated per request and thread it through transport/conversion/cancellation.
- Prefer small targeted fixes in this file; many routes share helper behavior.

## ANTI-PATTERNS
- Do not bypass `Depends(validate_api_contract)` or `Depends(validate_openai_api_contract)` on protected routes.
- Do not change SSE event ordering rules here without synchronized conversion/test updates.
- Do not remove request-id headers from error paths.

## VERIFICATION
```bash
python tests/test_main.py
python test_cancellation.py
```