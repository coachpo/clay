# Anthropic-to-OpenAI Tool Mapping Plan

## Summary

- Expand Clay's Anthropic surface to support the mentioned Anthropic-equivalent capabilities: structured outputs, web search, tool search, computer use, bash, text editor, code execution, and MCP toolsets.
- Keep the proxy Anthropic-native and strict. Accept only official Anthropic request shapes, gate each feature by the official `anthropic-version` that introduced it, and reject anything without a clean OpenAI equivalent.
- Do not add proxy-only support for OpenAI-only surfaces such as `allowed_tools`, generic `custom` raw-text tools, file search, image generation, or skills on Anthropic endpoints.

## Key Changes

- Add one central feature/version registry and one central OpenAI model-capability registry.
  - Validate requested Anthropic features against the resolved `anthropic-version`.
  - Validate mapped OpenAI model capabilities before conversion.
  - Treat unknown OpenAI model IDs conservatively: advanced features are rejected unless explicitly marked supported.
- Expand the Claude schema surface.
  - Extend the tool union with official Anthropic models for tool search, computer, bash, text editor, code execution, and MCP toolsets.
  - Extend content-block unions with the official built-in/server-tool blocks needed for those tools, including `tool_reference` and the documented server-tool result blocks.
  - Extend `output_config` to include structured-output fields in addition to `effort`.
  - Keep `tool_choice` limited to Anthropic's documented `auto|any|tool|none`.
- Expand request conversion.
  - Structured outputs: map Anthropic structured-output config to the current OpenAI Responses structured-output field; reject when the mapped model lacks support, especially `gpt-5.4-pro`.
  - Web search: map Anthropic `web_search_*` configuration to the OpenAI web-search tool; preserve max-uses and domain filters only where OpenAI has a documented equivalent, otherwise reject.
  - Tool search: map Anthropic tool-search tools to OpenAI tool-search and preserve tool-reference identity.
  - Computer use: map Anthropic computer tool declarations and actions to OpenAI computer use.
  - Bash, text editor, code execution: support only the documented overlapping subset with hosted shell, apply patch, and code interpreter; reject Anthropic-only operations that cannot be losslessly converted.
  - MCP: map Anthropic MCP toolsets to OpenAI MCP declarations; honor allow/deny/defer-loading only where there is a documented equivalent, otherwise reject.
- Expand response conversion.
  - Non-stream: convert OpenAI Responses output items for the new tool families into the corresponding Anthropic content blocks.
  - Stream: extend the SSE bridge to emit official Anthropic server-tool block lifecycle without breaking the existing Claude event order.
  - Never leak OpenAI-native item types on Anthropic routes; normalize or reject instead.
- Extend deterministic token counting.
  - Add estimator coverage for each new Anthropic tool and content-block family by serializing the official Anthropic payload shape or its documented text representation.
  - Replace current web-search rejection tests with coverage for accepted mappings and strict rejection of unsupported subfields.

## Delivery Order

1. Foundation: feature/version registry, OpenAI capability registry, and validation plumbing.
2. Schema: tool unions, content blocks, `output_config`, and role/tool-choice validators.
3. Request conversion: structured outputs, web search, tool search, then computer/bash/text editor/code execution/MCP.
4. Response conversion: non-stream first, then streaming SSE support for each new tool family.
5. Token counting, config/docs updates, and regression cleanup.

## Test Plan

- Parsing and validation:
  - Each new tool family parses on a supported `anthropic-version` and fails on an older one.
  - Structured outputs parse independently of reasoning and enforce model support correctly.
  - Existing Anthropic `tool_choice` rules still hold.
- Request conversion:
  - Structured outputs convert on `gpt-5.4` and fail on `gpt-5.4-pro` or unknown-capability models.
  - Web search, tool search, computer, bash, text editor, code execution, and MCP each produce the expected OpenAI Responses request shape.
  - Unsupported Anthropic-only fields fail fast with Anthropic-shaped 400s.
- Response conversion:
  - Non-stream and stream cases for each mapped tool family produce valid Anthropic content blocks.
  - Tool IDs, `tool_reference` identity, citations/result metadata, and request-id headers survive the round trip.
  - SSE ordering remains Claude-compatible for mixed text, thinking, and tool output streams.
- Regression:
  - Existing function-calling, reasoning, image/document input, context-management, cancellation, and model-discovery tests still pass.
  - `/v1/messages/count_tokens` remains deterministic and covers the new families.

## Assumptions

- Full scope means all mentioned Anthropic-equivalent tool families are planned now, but only through standard Anthropic request/response shapes.
- Strict compatibility means no Anthropic-surface support for OpenAI-only `allowed_tools`, generic `custom` tools, file search, image generation, or skills.
- Feature gating is enforced against the resolved `anthropic-version`, including fallback behavior.
- `ANTHROPIC_DEFAULT_VERSION` stays `2023-06-01`; advanced tools require newer configured supported versions rather than a silent default-version change.
- Raising `MAX_TOKENS_LIMIT` and `REQUEST_TIMEOUT` is recommended as a follow-up operational change, not part of the core mapping implementation.
