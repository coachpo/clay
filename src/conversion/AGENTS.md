# CONVERSION KNOWLEDGE BASE

## OVERVIEW
`conversion/` translates Claude request/response contracts to OpenAI-compatible payloads, including SSE bridge semantics.

## WHERE TO LOOK
- Request adaptation: `request_converter.py`.
- Response + streaming adaptation: `response_converter.py`.
- Tool result normalization: `parse_tool_result_content()`.
- Finish reason mapping: `_map_finish_reason()`.

## LOCAL CONTRACTS
- Request conversion clamps token budgets using config min/max limits.
- Reasoning-model requests (`o1/o3/o4/gpt-5`) use `max_completion_tokens` and may set `reasoning_effort` from thinking budget.
- Extension passthrough (`top_k`, `thinking`, `inference_geo`) is allowed only for non-OpenAI/non-Azure base URLs.
- Tool-call identities must survive round-trip conversion (`tool_use` <-> `tool_calls`).
- Streaming bridge must emit Claude SSE sequence: `message_start`, content block events, `message_delta`, `message_stop`.
- Tool-call argument deltas are reconstructed incrementally and buffered until tool block start.
- Usage extraction must tolerate provider shape variants (`prompt_tokens` vs `input_tokens`).

## CONVENTIONS
- Keep conversion deterministic and side-effect free except for logging.
- Preserve Unicode in JSON serialization (`ensure_ascii=False`) for content fidelity.
- Prefer explicit mapping helpers over inline conditional rewrites.

## ANTI-PATTERNS
- Do not emit OpenAI-native SSE events on Anthropic streaming routes.
- Do not collapse multiple content block types into plain text when block semantics matter.
- Do not change finish-reason mapping without updating integration checks.

## VERIFICATION
```bash
python tests/test_main.py
python test_cancellation.py
```