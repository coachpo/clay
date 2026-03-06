# API KNOWLEDGE BASE

## READ WHEN
- Editing `app/api/endpoints.py`.
- Changing request validation, route contracts, error envelopes, or cancellation flow.

## OWNED SURFACE
- Anthropic generation: `POST /v1/messages`.
- Anthropic token estimation: `POST /v1/messages/count_tokens`.
- OpenAI-compatible model discovery: `GET /v1/models`, `GET /v1/models/{model_id}`.
- Removed generation compatibility routes that must stay `404`: `POST /v1/chat/completions`, `POST /v1/responses`.
- Utility endpoints: `GET /health`, `GET /test-connection`, `GET /`.

## REQUEST CONTRACT GATES
- Anthropic route gate: `validate_api_contract` (size/version/content-type/client-key checks).
- OpenAI model-route gate: `validate_openai_api_contract` (Authorization/Bearer or `x-api-key`).
- Request-size limit is `MAX_ANTHROPIC_REQUEST_BYTES = 32 * 1024 * 1024` based on `content-length`.
- Anthropic version policy is resolved by `_resolve_requested_anthropic_version` with config-driven fallback behavior.
- Adaptive thinking is model-gated by `ADAPTIVE_THINKING_MODEL_PREFIXES`.

## REQUEST FLOW
- `create_message`: parse Claude request -> validate thinking/model compatibility -> convert -> dispatch stream/non-stream.
- Stream path uses `openai_client.create_response_stream` + `convert_openai_streaming_to_claude_with_cancellation`.
- Non-stream path uses `_create_response_with_disconnect_cancellation`; client disconnect returns `499` cancelled-style Anthropic error.
- Optional compatibility metadata may be injected into provider request via `_apply_anthropic_compat_enhancements` (`extra_body.proxy_metadata`).

## RESPONSE AND ERROR CONTRACTS
- Preserve `request-id` and `x-request-id` parity on success and error responses.
- Keep Anthropic and OpenAI error envelope shapes route-family specific.
- Keep status-to-error-type mapping helpers behavior consistent with test expectations.

## TOKEN COUNT ESTIMATION
- `/v1/messages/count_tokens` estimates from system/messages/tools/tool_choice/thinking/context payloads.
- Estimation uses `tiktoken` when available and falls back to heuristic counting.

## LOCAL ANTI-PATTERNS
- Do not bypass `Depends(validate_api_contract)` or `Depends(validate_openai_api_contract)`.
- Do not re-enable removed generation routes without synchronized converter/model/test work.
- Do not break request-id header parity or response envelope shape.
- Do not alter disconnect/cancellation semantics without updating stream and non-stream tests.

## VERIFICATION
```bash
python tests/test_main.py
```
