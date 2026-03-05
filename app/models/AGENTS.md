# MODELS KNOWLEDGE BASE

## OVERVIEW
`models/` defines strict Claude schema contracts and permissive OpenAI compatibility request models used by API validation and conversion flows.

## WHERE TO LOOK
- Claude request/content/tool/thinking schemas: `claude.py`.
- OpenAI compatibility request schemas: `openai.py`.
- Role/content compatibility: `ClaudeMessage.validate_role_content()`.
- Tool-choice/thinking coupling: `_ClaudeMessagesRequestFields.validate_tool_choice()`.
- Forward-compat parsing selectors: `parse_claude_messages_request()` and `parse_claude_token_count_request()`.

## LOCAL CONTRACTS
- Strict Claude models inherit `StrictBaseModel` (`extra="forbid"`) by default.
- Forward-compat Claude variants inherit `ForwardCompatBaseModel` (`extra="allow"`) when enabled by config flags.
- OpenAI compatibility models inherit `OpenAIBaseModel` (`extra="allow"`) and intentionally accept unknown fields.
- Claude content unions are discriminated by `type`; discriminator literals are protocol-level contract surface.
- Role enforcement: user blocks allow `text|image|document|tool_result`; assistant blocks allow `text|tool_use|thinking|redacted_thinking`.
- `thinking.budget_tokens` is valid only when `thinking.type == "enabled"`.
- `tool_choice` requires `tools` unless choice type is `none`; thinking only supports tool_choice `auto|none`.
- Context edit ordering is constrained (`clear_thinking_20251015` must be first when combining edits).
- Token-related numeric fields are bounded positive values where declared (for example `max_tokens`, `max_output_tokens`).

## CONVENTIONS
- When adding Claude block/tool variants, update schema unions, validators, request converter, response converter, and tests together.
- Keep defaults and field constraints aligned with endpoint behavior and integration assertions.
- Prefer schema-level validation over downstream ad-hoc runtime checks.

## ANTI-PATTERNS
- Do not loosen strict Claude models (`extra="forbid"`) without explicit contract decision.
- Do not add schema fields without corresponding API/conversion handling and regression tests.
- Do not assume OpenAI compatibility schemas are dead code; they define public compatibility surface expectations.

## VERIFICATION
```bash
mypy app
python tests/test_main.py
```
