# CORE KNOWLEDGE BASE

## OVERVIEW
`core/` owns import-time runtime configuration, provider transport and retries, cancellation primitives, model routing policy, constants, and shared logging setup.

## WHERE TO LOOK
- Env/config parsing + compatibility flags: `config.py`.
- Provider transport + optional-field fallback + cancellation map: `client.py`.
- Claude-to-target model routing policy: `model_manager.py`.
- Shared logging initialization and uvicorn noise suppression: `logging.py`.
- Protocol literals and event/stop-reason constants: `constants.py`.

## LOCAL CONTRACTS
- `Config()` requires `OPENAI_API_KEY`; invalid config exits process at import time (`sys.exit(1)`).
- `ANTHROPIC_API_KEY` is optional; absence disables client-key validation with a startup warning.
- `OPENAI_RESPONSES_STATE_MODE` must be `stateless` or `provider`.
- `OPENAI_GPT5_SAMPLING_REASONING_COMPAT_MODE` must be one of `off`, `drop_sampling`, `force_reasoning_none`, `strict_error`.
- `MIDDLE_MODEL` defaults to `BIG_MODEL` when unset.
- `OpenAIClient._normalize_base_url()` appends `/v1` only for non-Azure mode root URLs.
- Optional-field fallback retries once without `metadata`/`context_management` when upstream rejects those parameters.
- `OpenAIClient.active_requests[request_id]` stores cancellation events shared with API/conversion layers.
- `create_response_stream()` always forces `stream=True` on provider request.
- `cancel_request(request_id)` signals cancellation via `asyncio.Event` and must stay non-blocking.
- Model pass-through applies to prefix-matched names only (`gpt-*`, `o1-*`, `o3-*`, `o4-*`, `gpt-5`, `ep-*`, `doubao-*`, `deepseek-*`); other names are routed by `haiku`/`sonnet`/`opus` heuristics to configured models.

## CONVENTIONS
- Normalize provider failures to `HTTPException` with classified, user-actionable messages.
- Keep log-level parsing semantics aligned between `core/logging.py` and `app/main.py`.
- Treat `constants.py` as contract surface; avoid ad-hoc event/status strings in callers.

## ANTI-PATTERNS
- Do not move config validation from import-time to request-time without coordinated startup behavior changes.
- Do not break `request_id` cancellation semantics in client request lifecycle.
- Do not narrow model pass-through prefixes without explicit routing-policy decision.
- Do not remove fallback behavior for rejected optional provider fields without test updates.

## VERIFICATION
```bash
mypy app/
python tests/test_main.py
```
