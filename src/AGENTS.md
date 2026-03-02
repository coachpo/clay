# SRC KNOWLEDGE BASE

## READ WHEN
- Modifying any file under `src/**`.
- Investigating request translation, streaming/cancellation behavior, provider errors, or model routing.

## CHILD GUIDES
- `src/api/**` -> `src/api/AGENTS.md`
- `src/core/**` -> `src/core/AGENTS.md`
- `src/conversion/**` -> `src/conversion/AGENTS.md`
- `src/models/**` -> `src/models/AGENTS.md`
- Nearest child guide overrides this file for local details.

## MODULE MAP
- `main.py`: FastAPI app, exception-shape adapters, uvicorn launcher.
- `api/endpoints.py`: Anthropic `/v1/messages` + `/v1/messages/count_tokens`, OpenAI-compatible `/v1/models*`, contract validation, stream/non-stream dispatch.
- `core/config.py`: env loading, runtime defaults, compatibility flags, import-time validation gate.
- `core/client.py`: async provider transport, error mapping, cancellation primitives.
- `core/model_manager.py`: Claude model-name mapping rules.
- `conversion/request_converter.py`: Claude payload/tool/result conversion to OpenAI Responses format.
- `conversion/response_converter.py`: OpenAI responses/SSE conversion back to Claude format.
- `models/claude.py`: strict request schemas and role/content validators.
- `models/openai.py`: permissive OpenAI compatibility request models (not currently wired to active routes).

## CROSS-MODULE CONTRACTS
- Preserve request IDs in both headers: `request-id` and `x-request-id`.
- `/v1/messages` enforces request size, anthropic-version resolution, and JSON content type before provider calls.
- OpenAI-compatible generation routes are intentionally removed; only `/v1/models*` remains.
- Token ceilings are clamped through config limits (`MIN_TOKENS_LIMIT` and `MAX_TOKENS_LIMIT`).
- Tool-call IDs must stay stable between assistant `tool_use` and user `tool_result`.
- Claude streaming order is strict: `message_start` -> `ping` -> content events -> `message_delta` -> `message_stop`.
- Disconnect cancellation uses shared `request_id` across API handler, converter, and client map.

## LOCAL COMMANDS
```bash
uv run mypy src/
uv run black src/
uv run isort src/
uv run ruff check src/
python start_proxy.py
```

## GOTCHAS
- `core/config.py` and `core/logging.py` execute import-time side effects; avoid moving imports casually.
- `main.py` and `core/logging.py` each normalize log level; keep behavior aligned if touched.
- `start_proxy.py` mutates `sys.path`; prefer `uv run clay` for stable packaging path.
- `api/endpoints.py` is intentionally large and contract-heavy; keep bugfixes scoped and regression-tested.

## ESCALATION
- Changing API response/error shapes requires updating `tests/test_main.py` and `tests/AGENTS.md`.
- Changing startup/runtime command guidance requires updating root `AGENTS.md`.
