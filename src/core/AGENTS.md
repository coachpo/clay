# CORE KNOWLEDGE BASE

## OVERVIEW
`core/` contains runtime configuration, provider transport, logging setup, constants, and model routing policy.

## WHERE TO LOOK
- Config/env behavior and compatibility flags: `config.py`.
- Provider client + cancellation map: `client.py`.
- Claude-to-target model mapping: `model_manager.py`.
- Log-level normalization and logger wiring: `logging.py`.
- Shared protocol literals: `constants.py`.

## LOCAL CONTRACTS
- `Config()` requires `OPENAI_API_KEY`; invalid config exits process during module import.
- `OPENAI_RESPONSES_STATE_MODE` must be `stateless` or `provider`.
- `MIDDLE_MODEL` defaults to `BIG_MODEL` when unset.
- `OpenAIClient.active_requests` tracks cancellable requests by `request_id`.
- `create_response_stream()` always forces streaming mode on the provider request.
- `cancel_request(request_id)` signals cancellation via `asyncio.Event` and is relied on by API/conversion layers.
- Model mapping allows pass-through for `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `gpt-5`, `ep-*`, `doubao-*`, and `deepseek-*`.

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