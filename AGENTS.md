# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-05T05:52:05+02:00  
**Commit:** c2ee483  
**Branch:** main

## OVERVIEW

Clay is a Python/FastAPI compatibility proxy that serves Anthropic-compatible generation (`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI-compatible model discovery (`/v1/models*`).
Requests are translated into OpenAI Responses payloads with deterministic conversion, Claude-shape SSE bridging, request-id parity, and request-scoped cancellation.

## AGENTS HIERARCHY

- Nearest `AGENTS.md` wins; parent guidance still applies unless a child overrides it.
- Root rules apply everywhere.
- `app/**` -> `app/AGENTS.md`
- `app/api/**` -> `app/api/AGENTS.md`
- `app/core/**` -> `app/core/AGENTS.md`
- `app/conversion/**` -> `app/conversion/AGENTS.md`
- `app/models/**` -> `app/models/AGENTS.md`
- `tests/**` -> `tests/AGENTS.md`

## STRUCTURE

```text
clay/
|-- app/
|   |-- main.py                  # FastAPI app assembly + path-aware exception shaping + uvicorn launcher
|   |-- api/endpoints.py         # Anthropic generation contract + OpenAI models surface + validation/cancellation
|   |-- core/                    # config/env gates, provider transport/retries, logging, constants, model routing
|   |-- conversion/              # Claude<->OpenAI request/response translation + streaming event bridge
|   `-- models/                  # strict Claude schemas + permissive OpenAI compatibility schemas
|-- tests/test_main.py           # script-style integration harness (HTTP + converter/client checks)
|-- .github/workflows/           # CI quality gates, docker build/push, cleanup automation
|-- start_proxy.sh               # compatibility wrapper (`python -m app.main`)
|-- pyproject.toml               # packaging metadata + tooling config + `clay` entrypoint
`-- Dockerfile                   # production container image (non-root runtime)
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Anthropic request lifecycle | `app/api/endpoints.py` | `/v1/messages` validation, conversion dispatch, stream/non-stream flow |
| Token counting behavior | `app/api/endpoints.py` | `/v1/messages/count_tokens` estimation across message/tool/thinking blocks |
| OpenAI-compatible surface | `app/api/endpoints.py` | `/v1/models*` active; generation routes intentionally return 404 |
| Claude -> OpenAI conversion | `app/conversion/request_converter.py` | field validation, tool/thinking/context mapping, token clamp |
| OpenAI stream -> Claude SSE | `app/conversion/response_converter.py` | Responses + legacy chunk normalization with strict Claude event order |
| Provider transport + retries | `app/core/client.py` | optional-field fallback (`metadata/context_management/extra_body`) + protocol error mapping |
| Config/env behavior | `app/core/config.py` | import-time validation, compatibility flags, state-mode guard |
| Schema constraints | `app/models/claude.py`, `app/models/openai.py` | strict Claude validation vs permissive OpenAI request models |
| Integration behavior checks | `tests/test_main.py` | canonical contract harness and regression matrix |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
| --- | --- | --- | --- | --- |
| `create_message` | async function | `app/api/endpoints.py` | high | Main Anthropic `/v1/messages` lifecycle |
| `_create_response_with_disconnect_cancellation` | async function | `app/api/endpoints.py` | medium | Non-stream disconnect-aware cancellation |
| `OpenAIClient._create_with_metadata_fallback` | async method | `app/core/client.py` | high | Optional-field fallback retry policy |
| `OpenAIClient.create_response_stream` | async method | `app/core/client.py` | high | Streaming Responses transport + cancellation checks |
| `convert_claude_to_openai` | function | `app/conversion/request_converter.py` | high | Claude -> OpenAI Responses payload translation |
| `convert_openai_streaming_to_claude_with_cancellation` | async function | `app/conversion/response_converter.py` | high | OpenAI stream -> Claude SSE bridge with disconnect cancellation |
| `parse_claude_messages_request` | function | `app/models/claude.py` | medium | Strict vs forward-compat request parsing gate |
| `ModelManager.map_claude_model_to_openai` | method | `app/core/model_manager.py` | medium | Model routing policy |

## CONVENTIONS

- Use `pyproject.toml` as dependency/source-of-truth (`python -m pip install '.[dev]'` for local checks).
- Canonical runtime entrypoint is `clay` (`pyproject` script -> `app.main:main`); `start_proxy.sh` is compatibility-only.
- Contract responses include both `request-id` and `x-request-id` headers on Anthropic/OpenAI paths.
- Anthropic header validation is config-driven (`ANTHROPIC_SUPPORTED_VERSIONS`, fallback/missing toggles, default `2023-06-01`).
- Compatibility sampling fields (`temperature`, `top_p`) are accepted but never forwarded upstream.
- Provider fallback retries can strip `metadata`, `context_management`, and `extra_body` on unsupported/retryable/protocol failures.
- Canonical behavior verification is `python tests/test_main.py`; CI quality workflow runs static checks + compile smoke only.
- Docker image context is allowlist-based via `.dockerignore` (`app/**` + `pyproject.toml`).

## ANTI-PATTERNS (THIS PROJECT)

- Do not rely on stale docs command `python app/test_claude_to_openai.py` (file does not exist).
- Do not use removed generation routes `POST /v1/chat/completions` or `POST /v1/responses`; they intentionally return 404.
- Do not assume `docker-compose.yml` exists; this repo currently has no compose file.
- Do not remove request-id header parity (`request-id` and `x-request-id`) from API responses.
- Do not bypass contract validators on protected routes (`validate_api_contract`, `validate_openai_api_contract`).
- Do not reorder Claude SSE lifecycle events; clients depend on strict order.
- Do not treat provider-dependent integration skips/timeouts as deterministic product regressions.

## UNIQUE STYLES

- One router serves Anthropic-compatible generation and OpenAI-compatible model discovery with explicit contract separation.
- Conversion layer handles both Responses-style events and legacy chat-completion chunks while preserving Claude SSE semantics.
- Cancellation is coordinated across API, conversion, and client layers via shared `request_id`.
- Optional Anthropic compatibility metadata can be attached to provider payload `extra_body.proxy_metadata`.

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

## NOTES

- `tests/test_main.py` is the canonical integration harness and executes sequentially via `asyncio.run(main())`.
- `.github/workflows/ci.yml` enforces quality gates and docker smoke build, but does not run live integration tests.
- `.github/workflows/docker-images.yml` builds multi-arch images (`linux/amd64,linux/arm64`) and only pushes outside PRs.
- `config = Config()` runs at import time and exits on invalid required env or invalid `OPENAI_RESPONSES_STATE_MODE`.
- `README.md` is operator quickstart; AGENTS guides remain the implementation-focused reference.
