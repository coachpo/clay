# TESTS KNOWLEDGE BASE

## READ WHEN
- Editing files under `tests/**` (currently centered on `tests/test_main.py`).
- Verifying behavior changes in API, conversion, client fallback/cancellation, or schema contracts.

## CANONICAL HARNESS
- Primary behavior harness is `tests/test_main.py` (script-style, not pytest-first).
- Test execution is sequential via `asyncio.run(main())` with printed checkpoints.
- Suite combines live HTTP checks and in-process checks for converter/client helpers.

## RUNTIME ASSUMPTIONS
- Default target is `BASE_URL=http://localhost:8000` unless overridden.
- Environment is loaded from `.env` via `load_dotenv(PROJECT_ROOT / ".env")`.
- Several checks are provider-dependent and skip when `/test-connection` indicates upstream unavailability.

## COVERAGE HOTSPOTS
- Request-id header parity (`request-id == x-request-id`) on success and error paths.
- Removed endpoint behavior (`/v1/chat/completions`, `/v1/responses` -> `404`).
- Optional-field fallback retries for `metadata`/`context_management`/`extra_body`.
- Sampling-ignore behavior (`temperature`, `top_p` not forwarded).
- Reasoning effort resolution behavior and omission conditions.
- Streaming event ordering/tool reasoning handling in conversion bridge.

## WORKFLOW REALITY
- CI quality workflow does not run this integration script.
- `pyproject.toml` contains pytest config, but repo contract checks are still centered on this script.

## LOCAL ANTI-PATTERNS
- Do not assume provider/network failures are always proxy regressions.
- Do not change API/converter contracts without updating assertions in this file in the same change.
- Do not revert unrelated user edits while refreshing AGENTS guidance.

## COMMAND
```bash
python tests/test_main.py
```
