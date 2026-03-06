# PROJECT KNOWLEDGE BASE

## OVERVIEW
Clay is a FastAPI compatibility proxy that accepts Anthropic-style generation requests and forwards them to OpenAI-compatible Responses providers.
It preserves Claude request/response contracts (including SSE event ordering), maintains request-id header parity, and supports request-scoped cancellation.

## AGENTS HIERARCHY
- Nearest `AGENTS.md` wins; parent guidance still applies unless a child overrides it.
- Root rules apply everywhere.
- `app/**` -> `app/AGENTS.md`
- `app/api/**` -> `app/api/AGENTS.md`
- `app/core/**` -> `app/core/AGENTS.md`
- `app/conversion/**` -> `app/conversion/AGENTS.md`
- `app/models/**` -> `app/models/AGENTS.md`
- `tests/**` -> `tests/AGENTS.md`

## CURRENT API SURFACE
- Anthropic generation: `POST /v1/messages`
- Anthropic token estimation: `POST /v1/messages/count_tokens`
- OpenAI-compatible model discovery: `GET /v1/models`, `GET /v1/models/{model_id}`
- Removed generation compatibility routes: `POST /v1/chat/completions`, `POST /v1/responses` (intentionally `404`)
- Utility routes: `GET /health`, `GET /test-connection`, `GET /`

## STRUCTURE MAP
```text
clay/
|-- app/
|   |-- main.py                  # FastAPI app wiring + exception shaping + CLI entrypoint
|   |-- api/endpoints.py         # route handlers, auth/version/content-type gates, cancellation glue
|   |-- core/                    # config/env validation, transport/retry/cancel, routing, logging, constants
|   |-- conversion/              # Claude <-> OpenAI request/response conversion + streaming bridge
|   `-- models/                  # strict Claude schemas + permissive OpenAI compatibility schemas
|-- tests/test_main.py           # script-style integration harness (HTTP + in-process checks)
|-- .github/workflows/           # CI quality gates + Docker image workflows + cleanup
|-- start_proxy.sh               # compatibility wrapper (`python -m app.main`)
|-- pyproject.toml               # dependency/tooling source of truth + `clay` script entrypoint
`-- Dockerfile                   # production container image (non-root runtime)
```

## CORE CONTRACTS (DO NOT BREAK)
- Preserve header parity on contract routes: both `request-id` and `x-request-id`.
- Keep Anthropic contract validators on protected routes (`validate_api_contract`, `validate_openai_api_contract`).
- Enforce Anthropic request size limit at `32 MB` (`MAX_ANTHROPIC_REQUEST_BYTES`) using `content-length`.
- Keep Anthropic version handling config-driven (`ANTHROPIC_*` flags; default `2023-06-01`).
- Keep sampling compatibility behavior: accept `temperature` and `top_p`, but never forward upstream.
- Keep OpenAI optional-field fallback behavior in client retries for `metadata`, `context_management`, and `extra_body`.
- Keep streaming SSE lifecycle order: `message_start` -> `ping` -> block events -> `message_delta` -> `message_stop`.
- Keep cancellation wired by shared `request_id` across API handlers, converters, and `OpenAIClient.active_requests`.

## REASONING EFFORT RESOLUTION
- First source: `output_config.effort` (`low -> medium`, `medium -> high`, `high|max -> xhigh`).
- Fallback source: `thinking` budget (`enabled` with `<1024 -> medium`, `<4096 -> high`, `>=4096 -> xhigh`).
- Omit `reasoning` when `thinking.type` is `adaptive` or `disabled`, or when no effort resolves.
- Adaptive thinking is model-gated by prefixes in `ADAPTIVE_THINKING_MODEL_PREFIXES` (`claude-opus-4-6`, `claude-sonnet-4-6`).

## TESTING AND TOOLING REALITY
- Canonical behavior harness: `python tests/test_main.py` (script-style, `asyncio.run(main())`).
- CI (`.github/workflows/ci.yml`) runs formatting, lint, type checks, and compile smoke; it does not run `python tests/test_main.py`.
- Runtime entrypoint is `clay` (`pyproject.toml` -> `app.main:main`); `start_proxy.sh` is a compatibility wrapper.
- Docker build context is allowlist-based via `.dockerignore` (runtime files only).

## COMMANDS
```bash
python -m pip install --upgrade pip
python -m pip install '.[dev]'
clay
./start_proxy.sh
isort --check-only app tests
black --check app tests
ruff check app tests
mypy app
python -m compileall -q app tests
python tests/test_main.py
docker build -t clay .
```

## ANTI-PATTERNS (THIS REPO)
- Do not re-enable `POST /v1/chat/completions` or `POST /v1/responses` as generation paths without full contract/conversion/test updates.
- Do not remove request-id header parity or route-specific error envelope shapes.
- Do not bypass schema/contract validation for convenience in handlers.
- Do not reorder Claude SSE lifecycle events.
- Do not assume provider-dependent skips/timeouts in `tests/test_main.py` are deterministic product regressions.
- Do not rely on nonexistent legacy command `python app/test_claude_to_openai.py`.

## CHANGE CHECKLIST
- Update relevant child `AGENTS.md` when local behavior changes.
- Keep README operator docs and AGENTS implementation guidance aligned.
- For API/conversion behavior changes, update `tests/test_main.py` in the same change.
