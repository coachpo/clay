# CORE KNOWLEDGE BASE

## OVERVIEW
`core/` contains runtime configuration, provider transport, logging setup, constants, and model routing policy.

## WHERE TO LOOK
- Config/env behavior: `config.py`.
- Provider client + cancellation map: `client.py`.
- Claude-to-target model mapping: `model_manager.py`.
- Log-level normalization and logger wiring: `logging.py`.
- Shared protocol literals: `constants.py`.

## LOCAL CONTRACTS
- `config = Config()` executes at import time and exits process on invalid required env.
- `MIDDLE_MODEL` defaults to `BIG_MODEL` when unset.
- `get_custom_headers()` maps `CUSTOM_HEADER_*` env vars to hyphenated HTTP headers.
- `OpenAIClient.active_requests` tracks cancellable requests by `request_id`.
- Streaming calls force `stream_options.include_usage = True` before provider invocation.
- `cancel_request(request_id)` signals cancellation via `asyncio.Event` and is relied on by API/conversion layers.
- Model mapping allows pass-through for `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `gpt-5`, `ep-*`, `doubao-*`, `deepseek-*`.

## CONVENTIONS
- Keep provider errors normalized through `HTTPException` with classified messages.
- Keep log-level parsing behavior consistent between `core/logging.py` and `main.py`.
- Treat constants as protocol contract surface; avoid ad-hoc string literals in callers.

## ANTI-PATTERNS
- Do not defer config validation from import-time to request-time without coordinated startup changes.
- Do not break `request_id` cancellation semantics in client methods.
- Do not narrow model pass-through prefixes without explicit routing-policy intent.

## VERIFICATION
```bash
uv run mypy src/
python tests/test_main.py
```