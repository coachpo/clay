# CONVERSION KNOWLEDGE BASE

## OVERVIEW
`conversion/` is the protocol bridge between Claude contracts and OpenAI provider payloads/events, including deterministic request mapping and Claude SSE-compliant streaming reconstruction.

## WHERE TO LOOK
- Claude -> OpenAI request conversion: `request_converter.py` (`convert_claude_to_openai`).
- OpenAI -> Claude non-stream conversion: `response_converter.py` (`convert_openai_to_claude_response`).
- OpenAI stream -> Claude SSE bridge: `convert_openai_streaming_to_claude_with_cancellation()`.
- Tool-result normalization: `parse_tool_result_content()`.
- Stop-reason mapping: `_map_finish_reason()` and `_map_responses_stop_reason()`.

## LOCAL CONTRACTS
- Request conversion clamps output token budgets with config min/max bounds.
- Responses payload `store` flag is derived from `OPENAI_RESPONSES_STATE_MODE` (`provider` vs `stateless`).
- Responses-only mode rejects unsupported Claude fields: `stop_sequences`, `top_k`, `inference_geo`.
- Native Claude `web_search*` tool variants are rejected for Responses-only conversion.
- Claude context edits are mapped to Responses `context_management` compaction entries.
- Tool-call identity must survive round trip (`tool_use.id` <-> OpenAI `call_id`/`id`).
- Streaming bridge accepts both Responses-style events and legacy chat chunk shapes.
- Claude streaming sequence is strict: `message_start` -> `ping` -> content block events -> `message_delta` -> `message_stop`.
- Tool-call JSON argument deltas are buffered/emitted incrementally and flushed when tool block starts/completes.
- Usage extraction tolerates provider variants (`prompt_tokens`/`input_tokens`, `completion_tokens`/`output_tokens`) and cache/reasoning detail fields.
- Sampling passthrough compatibility is universal: accept `temperature`/`top_p` for compatibility, but always drop them before upstream dispatch.

## STREAMING STATE MACHINE
- Emit `message_start`, then `ping`, before any block events.
- Create block indexes lazily when first text/thinking/tool data arrives.
- Buffer tool argument fragments until tool block metadata is ready; then emit `input_json_delta` fragments in order.
- Close all started blocks before final `message_delta` + `message_stop`.
- On disconnect/cancellation, emit Claude-shape error payload (`cancelled`) instead of partial OpenAI-native events.

## CONVENTIONS
- Keep conversions deterministic and side-effect free except logging warnings for compatibility shims.
- Preserve Unicode fidelity in serialized JSON (`ensure_ascii=False`).
- Prefer dedicated helper mappers over inline branching across request/stream code paths.

## ANTI-PATTERNS
- Do not emit OpenAI-native stream events on Anthropic streaming routes.
- Do not flatten semantic block types (`thinking`, `tool_use`, `tool_result`) into plain text.
- Do not alter stop-reason mapping semantics without synchronized integration-test updates.
- Do not reorder Claude SSE lifecycle events; clients depend on ordering.

## VERIFICATION
```bash
python tests/test_main.py
```
