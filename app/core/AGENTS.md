# CORE KNOWLEDGE BASE

## READ WHEN
- Editing anything in `app/core/**` (`config.py`, `client.py`, `model_manager.py`, `logging.py`, `constants.py`).
- Changing transport, retries, cancellation, runtime config, or model routing behavior.

## OWNED MODULES
- `config.py`: environment parsing, defaults, and startup validation.
- `client.py`: OpenAI/Azure transport, fallback retries, protocol error normalization, cancellation map.
- `model_manager.py`: Claude-to-target model mapping strategy.
- `logging.py`: shared logger setup and uvicorn log noise control.
- `constants.py`: protocol literals used across API/conversion layers.

## RUNTIME CONTRACTS
- `config = Config()` executes at import time; invalid config exits process (`sys.exit(1)`).
- `OPENAI_API_KEY` is required; `ANTHROPIC_API_KEY` is optional (missing key disables client-key validation).
- `OPENAI_RESPONSES_STATE_MODE` must be `stateless` or `provider`.
- `MIDDLE_MODEL` defaults to `BIG_MODEL` when unset.
- `_normalize_base_url` appends `/v1` only for non-Azure-style usage.

## CLIENT BEHAVIOR CONTRACTS
- `OpenAIClient.active_requests[request_id]` stores cancellation events for both stream and non-stream flows.
- `cancel_request(request_id)` must remain non-blocking and only signal cancellation.
- `create_response_stream()` always forces `stream=True` before upstream call.
- `_create_with_metadata_fallback()` retries once after stripping optional fields `metadata`, `context_management`, and `extra_body` when appropriate.
- Retry triggers include unsupported optional parameters, retryable gateway/server statuses, and protocol parse error signatures.

## ERROR NORMALIZATION
- Protocol mismatch/parsing failures are surfaced as normalized upstream protocol errors.
- Keep status-code normalization and user-facing classification stable (`classify_openai_error`).
- Keep JSON/non-JSON detection helpers aligned with current gateway behavior assumptions.

## MODEL ROUTING RULES
- Prefix pass-through models stay pass-through: `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `gpt-5`, `ep-*`, `doubao-*`, `deepseek-*`.
- Non-pass-through names route by substring heuristic (`haiku` -> small, `sonnet` -> middle, `opus` -> big, fallback big).

## LOCAL ANTI-PATTERNS
- Do not move config validation from import-time to request-time without explicit startup contract changes.
- Do not break `request_id` cancellation wiring across API/conversion/client.
- Do not remove optional-field fallback logic without synchronized test updates.
- Do not change pass-through prefixes casually; it is routing policy.

## VERIFICATION
```bash
mypy app
python tests/test_main.py
```
