# TESTS KNOWLEDGE BASE

## READ WHEN
- Editing anything under `tests/**`.
- Verifying behavior changes in `app/api`, `app/core/client`, `app/conversion`, or schema contracts.

## TEST LAYOUT
- Canonical harness: `tests/test_main.py` (script-style integration checks).
- Executes sequentially via `asyncio.run(main())` and prints progress per scenario.
- Contains both HTTP integration scenarios and in-process/unit-like converter/client checks.

## LOCAL CONTRACTS
- Default proxy target is `http://localhost:8000` unless `BASE_URL` is set.
- Environment is loaded from `.env` via `load_dotenv(PROJECT_ROOT / ".env")`.
- Header-parity checks assert `request-id == x-request-id` across success and error paths.
- Many network/provider-dependent checks can skip when upstream is unavailable (health/connectivity guards).
- Test matrix includes removed-endpoint assertions (`/v1/chat/completions`, `/v1/responses` -> 404).
- Coverage includes sampling-ignore behavior, context_management mapping, and streaming event-order semantics.
- Pytest config exists in `pyproject.toml`, but primary repo workflow remains `python tests/test_main.py`.

## LOCAL COMMANDS
```bash
python tests/test_main.py
```

## GOTCHAS
- Failures may be environment/provider related (`OPENAI_API_KEY`, reachability, rate limits) rather than proxy logic regressions.
- CI quality workflow does not execute this live integration script.
- README exists for quickstart; AGENTS files are still the operational implementation reference.

## ESCALATION
- If API/converter contract changes, update `tests/test_main.py` assertions in the same change.
- If introducing dedicated pytest suites/fixtures as primary flow, update root/test AGENTS command guidance accordingly.
