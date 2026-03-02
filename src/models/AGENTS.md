# MODELS KNOWLEDGE BASE

## OVERVIEW
`models/` defines strict Claude schemas and permissive OpenAI compatibility models used for request validation.

## WHERE TO LOOK
- Claude contracts + validators: `claude.py`.
- OpenAI compatibility model shapes: `openai.py`.
- Role/content compatibility checks: `ClaudeMessage.validate_role_content()`.
- Tool-choice/thinking constraints: `_ClaudeMessagesRequestFields.validate_tool_choice()`.

## LOCAL CONTRACTS
- Claude models inherit `StrictBaseModel` (`extra="forbid"`) to reject unknown fields.
- Forward-compat Claude variants use `ForwardCompatBaseModel` (`extra="allow"`) when enabled by config.
- OpenAI models inherit `OpenAIBaseModel` (`extra="allow"`) for permissive compatibility shapes.
- Claude message content uses discriminated unions by `type`; discriminator values must stay stable.
- `tool_result` blocks are user-only; `tool_use`/thinking blocks are assistant-only.
- `thinking.budget_tokens` is valid only when `thinking.type` is `enabled`.
- `tool_choice` requires `tools` unless choice type is `none`.
- Context edit order is constrained (`clear_thinking_20251015` must be first when combining edits).
- OpenAI request token fields are positive integers when present.

## CONVENTIONS
- When adding a Claude content block type, update unions, validators, and converters together.
- Keep model field defaults aligned with endpoint behavior and tests.
- Prefer schema-level constraints over ad-hoc runtime checks when feasible.

## ANTI-PATTERNS
- Do not loosen Claude strictness (`extra="forbid"`) without explicit contract decision.
- Do not add model-only fields without corresponding conversion/endpoint handling.
- Do not assume OpenAI schema layer is unused; compatibility models are part of documented API surface.

## VERIFICATION
```bash
uv run mypy src/
python tests/test_main.py
```