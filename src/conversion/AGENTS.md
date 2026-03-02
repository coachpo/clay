# CONVERSION KNOWLEDGE BASE

## OVERVIEW
`conversion/` translates Claude request/response contracts to OpenAI Responses payloads and back, including SSE bridge semantics.

## WHERE TO LOOK
- Request adaptation: `request_converter.py`.
- Response + streaming adaptation: `response_converter.py`.
- Tool result normalization: `parse_tool_result_content()`.
- Stop-reason mapping: `_map_finish_reason()` and `_map_responses_stop_reason()`.

## LOCAL CONTRACTS
- Request conversion clamps token budgets using config min/max limits.
- Requests map to OpenAI Responses payloads with `store` driven by `OPENAI_RESPONSES_STATE_MODE`.
- Unsupported Claude fields (`stop_sequences`, `top_k`, `inference_geo`, `context_management`) raise conversion errors in Responses-only mode.
- Native Claude `web_search` tool variants are rejected in Responses-only mode.
- Tool-call identities must survive round-trip conversion (`tool_use` <-> function call IDs).
- Streaming bridge emits Claude SSE sequence: `message_start`, `ping`, content events, `message_delta`, `message_stop`.
- Tool-call argument deltas are reconstructed incrementally and buffered until tool block start.
- Usage extraction tolerates provider shape variants (`prompt_tokens` vs `input_tokens`, `completion_tokens` vs `output_tokens`).

## CONVENTIONS
- Keep conversion deterministic and side-effect free except for logging.
- Preserve Unicode in JSON serialization (`ensure_ascii=False`) for content fidelity.
- Prefer explicit mapping helpers over inline conditional rewrites.

## ANTI-PATTERNS
- Do not emit OpenAI-native SSE events on Anthropic streaming routes.
- Do not collapse multiple content block types into plain text when block semantics matter.
- Do not change stop/finish-reason mapping without updating integration checks.

## VERIFICATION
```bash
python tests/test_main.py
```
