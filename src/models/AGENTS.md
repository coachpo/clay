# MODELS KNOWLEDGE BASE

## OVERVIEW
`models/` defines strict Claude schemas and permissive OpenAI-compatible request models for endpoint validation.

## WHERE TO LOOK
- Claude contracts + validators: `claude.py`.
- OpenAI request envelopes: `openai.py`.
- Role/content compatibility checks: `ClaudeMessage.validate_role_content()`.
- Tool-choice/thinking constraints: `ClaudeMessagesRequest.validate_tool_choice()`.

## LOCAL CONTRACTS
- Claude models inherit `StrictBaseModel` (`extra="forbid"`) to reject unknown fields.
- OpenAI models inherit `OpenAIBaseModel` (`extra="allow"`) for permissive compatibility.
- Claude message content uses discriminated unions by `type`; keep discriminator values stable.
- `tool_result` blocks are user-only; `tool_use`/thinking blocks are assistant-only.
- `thinking.budget_tokens` is valid only when thinking type is `enabled`.
- `tool_choice` requires `tools` unless choice type is `none`.
- OpenAI requests validate token fields as positive integers when present.

## CONVENTIONS
- When adding a new Claude content block type, update unions, validators, and converters together.
- Keep model field defaults aligned with endpoint behavior and docs.
- Prefer schema-level constraints over ad-hoc runtime checks when feasible.

## ANTI-PATTERNS
- Do not loosen Claude strictness (`extra="forbid"`) without explicit contract decision.
- Do not add model-only fields without corresponding conversion/endpoint handling.
- Do not assume OpenAI schema layer is unused; API routes import these request models directly.

## VERIFICATION
```bash
uv run mypy src/
python tests/test_main.py
```