# MODELS KNOWLEDGE BASE

## READ WHEN
- Editing `app/models/claude.py` or `app/models/openai.py`.
- Changing request schema validation or compatibility model acceptance behavior.

## OWNED SCHEMA SURFACE
- `claude.py`: strict Anthropic request/content/tool/thinking/context schemas + forward-compat variants.
- `openai.py`: permissive OpenAI compatibility request models (`extra="allow"`).
- Parse selectors: `parse_claude_messages_request` and `parse_claude_token_count_request`.

## CLAUDE SCHEMA CONTRACTS
- Strict models default to `extra="forbid"` via `StrictBaseModel`.
- Forward-compat models default to `extra="allow"` via `ForwardCompatBaseModel` and are selected by config flags.
- Content unions are discriminator-based by `type`; literals are part of public contract.
- Role/content rules are enforced in `ClaudeMessage.validate_role_content()`.
- `thinking.budget_tokens` is only valid when `thinking.type == "enabled"`.
- `tool_choice` validation enforces tools presence (except `none`) and thinking-compatible choice set (`auto|none`).
- `context_management` edit ordering enforces `clear_thinking_20251015` first when combining edits.

## OPENAI COMPATIBILITY MODEL CONTRACTS
- OpenAI request schemas intentionally allow unknown fields (`OpenAIBaseModel`).
- Compatibility models are still contract-relevant even when API generation routes are removed.

## CHANGE GUIDELINES
- When adding/changing block or tool variants, update schema unions + converter logic + tests together.
- Keep field bounds/defaults aligned with endpoint and converter assumptions.
- Prefer schema validators over ad-hoc runtime checks in API handlers.

## LOCAL ANTI-PATTERNS
- Do not loosen strict Claude schemas without an explicit contract decision.
- Do not add schema fields without corresponding conversion and regression-test coverage.
- Do not treat compatibility models as dead code; they document accepted surface area.

## VERIFICATION
```bash
mypy app
python tests/test_main.py
```
