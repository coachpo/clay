# APP KNOWLEDGE BASE

## READ WHEN
- Modifying any file under `app/**`.
- Coordinating behavior spanning API + conversion + client + model schemas.

## CHILD GUIDES
- `app/api/**` -> `app/api/AGENTS.md`
- `app/core/**` -> `app/core/AGENTS.md`
- `app/conversion/**` -> `app/conversion/AGENTS.md`
- `app/models/**` -> `app/models/AGENTS.md`
- Nearest child guide overrides this file for local details.

## MODULE MAP
- `main.py`: FastAPI app assembly, path-aware exception handlers, `clay` CLI runtime entrypoint.
- `api/endpoints.py`: Anthropic generation routes + OpenAI-compatible model discovery + request-contract gates.
- `core/config.py`: import-time env/config validation and compatibility feature flags.
- `core/client.py`: OpenAI/Azure transport, optional-field fallback retry (`metadata/context_management/extra_body`), protocol error normalization, active-request cancellation map.
- `core/model_manager.py`: Claude model-name routing to configured target providers/models.
- `core/logging.py`: shared logging config and uvicorn log-level suppression.
- `core/constants.py`: protocol constants for roles/content blocks/SSE events/stop reasons.
- `conversion/request_converter.py`: Claude request/tool/thinking/context conversion to OpenAI Responses payload.
- `conversion/response_converter.py`: Responses + legacy chunk normalization back to Claude JSON/SSE contracts.
- `models/claude.py`: strict Claude schemas + forward-compat variants + role/block validators.
- `models/openai.py`: permissive OpenAI compatibility request models (`extra="allow"`).

## CROSS-MODULE CONTRACTS
- Preserve request IDs in both headers on Anthropic/OpenAI paths: `request-id` and `x-request-id`.
- `/v1/messages` enforces size/version/content-type/auth gates before conversion/provider calls.
- OpenAI generation compatibility routes are intentionally removed (`/v1/chat/completions`, `/v1/responses` -> 404).
- `/v1/models*` remains active and is guarded by OpenAI-compatible API-key validation.
- Cancellation uses shared `request_id` across API handler, conversion stream bridge, and `OpenAIClient.active_requests`.
- Streaming event order must remain Claude-compatible: `message_start` -> `ping` -> block events -> `message_delta` -> `message_stop`.
- Sampling compatibility fields (`temperature`, `top_p`) are accepted but dropped before upstream dispatch.
- Provider protocol parse failures are normalized in core client and surfaced through API-specific error envelopes.

## LOCAL COMMANDS
```bash
python -m pip install --upgrade pip
python -m pip install '.[dev]'
mypy app
ruff check app
black --check app
isort --check-only app
clay
./start_proxy.sh
```

## GOTCHAS
- `core/config.py` and `core/logging.py` run import-time side effects; moving imports can change startup behavior.
- `main.py` and `core/logging.py` each normalize log levels; keep semantics aligned when changing either file.
- `api/endpoints.py` is large and contract-heavy; keep bugfixes minimal and targeted.
- Optional-field fallback in `core/client.py` is shared by stream + non-stream paths; do not split semantics casually.

## ESCALATION
- API response/error-shape changes require synchronized updates to converters + `tests/test_main.py` expectations.
- Startup/runtime command changes require updating root `AGENTS.md` command guidance.
