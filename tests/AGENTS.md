# TESTS KNOWLEDGE BASE

## READ WHEN
- Editing `tests/**`.
- Editing `test_cancellation.py`.
- Validating behavior changes in `src/api`, `src/core/client`, or `src/conversion`.

## TEST LAYOUT
- `tests/test_main.py`: broad integration scenarios across Anthropic and OpenAI-compatible routes.
- `../test_cancellation.py`: cancellation/disconnect checks plus OpenAI auth sanity check.
- Both scripts are executable entrypoints via `asyncio.run(main())`.

## LOCAL CONTRACTS
- Tests expect proxy at `http://localhost:8082` unless `BASE_URL` overrides it.
- Scripts load environment from `.env` via `load_dotenv()`.
- Many checks are provider-dependent and skip when upstream is unavailable/timeouts occur.
- Assertions validate request ID header parity (`request-id` matches `x-request-id`).
- Coverage is scenario-driven integration, not fixture-heavy pytest units.

## LOCAL COMMANDS
```bash
python tests/test_main.py
python test_cancellation.py
```

## GOTCHAS
- No `README.md` or `QUICKSTART.md` docs are maintained in this repository.
- No `pytest.ini`, `conftest.py`, or repo-level pytest options are present.
- Failures are often environment/provider issues (`OPENAI_API_KEY`, reachability, rate limits), not pure code regressions.

## ESCALATION
- If endpoint contracts change, update both scripts to keep Anthropic/OpenAI assertions aligned.
- If introducing real pytest suites/fixtures, update root `AGENTS.md` command guidance.
