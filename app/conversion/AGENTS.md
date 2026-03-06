# CONVERSION KNOWLEDGE BASE

## READ WHEN
- Editing `app/conversion/request_converter.py` or `app/conversion/response_converter.py`.
- Changing request mapping, stop-reason semantics, streaming event generation, or tool/thinking handling.

## OWNED RESPONSIBILITIES
- `convert_claude_to_openai` (alias) -> canonical Responses payload conversion.
- `convert_openai_to_claude_response` for non-stream mapping.
- `convert_openai_streaming_to_claude_with_cancellation` for Anthropic SSE stream output.
- Tool-result normalization and content extraction helpers used by API/token-count paths.

## REQUEST-CONVERTER CONTRACTS
- Unsupported fields for Responses-only mode must be rejected (`stop_sequences`, `top_k`, `inference_geo`).
- Native Claude `web_search*` tools are rejected in Responses-only mode.
- `store` is derived from `OPENAI_RESPONSES_STATE_MODE` (`provider` vs `stateless`).
- `temperature` and `top_p` are accepted for compatibility but always dropped upstream.
- Context edits map into Responses `context_management` compaction entries (threshold constant-driven).
- Reasoning effort resolution order is deterministic: `output_config.effort` -> `thinking` budget -> omit `reasoning`.

## RESPONSE/STREAM-CONVERTER CONTRACTS
- Stream bridge must preserve Claude lifecycle order: `message_start` -> `ping` -> block events -> `message_delta` -> `message_stop`.
- Bridge handles both Responses-style event streams and legacy chat-completion chunk streams.
- Tool call identity mapping must remain stable (`tool_use.id` <-> OpenAI call IDs/item IDs).
- Tool argument deltas are buffered/emitted in order; missing fragments are flushed on completion events.
- Usage extraction tolerates provider variants (`prompt_tokens`/`input_tokens`, `completion_tokens`/`output_tokens`, cache/reasoning detail fields).
- Disconnect/cancellation emits Claude-style `error` event with `type: cancelled` instead of OpenAI-native partial output.

## STOP-REASON MAPPING
- `_map_finish_reason` and `_map_responses_stop_reason` define contract semantics for `tool_use`, `max_tokens`, `stop_sequence`, `refusal`, `error`, and default end-turn.
- Any mapping change requires synchronized test updates in `tests/test_main.py`.

## LOCAL ANTI-PATTERNS
- Do not emit OpenAI-native event shapes to Anthropic streaming clients.
- Do not flatten semantic blocks (`thinking`, `tool_use`, `tool_result`) into plain text.
- Do not reorder SSE lifecycle events.
- Do not make nondeterministic conversion behavior changes.

## VERIFICATION
```bash
python tests/test_main.py
```
