# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-03T19:51:49+02:00  
**Commit:** 0057a2c  
**Branch:** main

## OVERVIEW

Clay is a Python/FastAPI compatibility proxy that serves Anthropic-compatible generation (`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI-compatible model discovery (`/v1/models*`).
Generation requests are translated to OpenAI Responses API payloads with deterministic conversion, SSE event bridging, request-id parity, and request-scoped cancellation wiring.

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
|   |-- main.py                  # FastAPI app + exception-shape adapters + uvicorn launcher
|   |-- api/endpoints.py         # Anthropic/OpenAI route contracts + validation/cancellation glue
|   |-- core/                    # config/env gates, provider client, logging, constants, model routing
|   |-- conversion/              # Claude<->OpenAI request/response translation + SSE bridge
|   `-- models/                  # strict Claude schema + permissive OpenAI compatibility schema
|-- tests/test_main.py           # script-style integration scenario runner
|-- .github/workflows/           # CI quality checks, docker build/push, cleanup automation
|-- start_proxy.sh               # compatibility wrapper (`python -m app.main`)
|-- pyproject.toml               # packaging metadata + tooling configuration + `clay` entrypoint
`-- Dockerfile                   # production container build (non-root runtime)
```

## WHERE TO LOOK

| Task                              | Location                                       | Notes                                                                 |
| --------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| Anthropic request lifecycle       | `app/api/endpoints.py`                         | `/v1/messages` validation, conversion dispatch, stream/non-stream flow |
| Token counting behavior           | `app/api/endpoints.py`                         | `/v1/messages/count_tokens` estimation and content-block tokenization  |
| OpenAI-compatible surface         | `app/api/endpoints.py`                         | `/v1/models*` active; generation routes intentionally return 404       |
| Request conversion                | `app/conversion/request_converter.py`          | Claude blocks/tools/thinking -> OpenAI Responses payload               |
| Stream conversion                 | `app/conversion/response_converter.py`         | OpenAI chunk/SSE events -> Claude SSE event order                      |
| Provider transport + cancellation | `app/core/client.py`                           | async OpenAI calls + `active_requests[request_id]` cancellation map    |
| Config/env behavior               | `app/core/config.py`                           | import-time validation, compatibility flags, state-mode guard          |
| Schema constraints                | `app/models/claude.py`, `app/models/openai.py` | strict Claude validation vs permissive OpenAI model shapes             |
| Integration behavior checks       | `tests/test_main.py`                           | script-driven end-to-end contract checks                               |

## CODE MAP

| Symbol                                                 | Type           | Location                               | Refs   | Role                                      |
| ------------------------------------------------------ | -------------- | -------------------------------------- | ------ | ----------------------------------------- |
| `create_message`                                       | async function | `app/api/endpoints.py`                 | high   | Main Anthropic request lifecycle          |
| `_create_response_with_disconnect_cancellation`        | async function | `app/api/endpoints.py`                 | medium | Non-stream disconnect cancellation        |
| `OpenAIClient.create_response_stream`                  | async method   | `app/core/client.py`                   | high   | Streaming Responses provider transport    |
| `convert_claude_to_openai`                             | function       | `app/conversion/request_converter.py`  | high   | Claude -> OpenAI payload translation      |
| `convert_openai_streaming_to_claude_with_cancellation` | async function | `app/conversion/response_converter.py` | high   | OpenAI stream -> Claude SSE bridge        |
| `ClaudeMessagesRequestModel`                           | type alias     | `app/models/claude.py`                 | medium | Union of strict and forward-compat Claude request schemas |
| `ModelManager.map_claude_model_to_openai`              | method         | `app/core/model_manager.py`            | medium | Model routing policy                      |

## CONVENTIONS

- Use `pyproject.toml` as dependency/source-of-truth (`python -m pip install '.[dev]'` for local quality checks).
- Canonical runtime entrypoint is `clay` (`pyproject` script -> `app.main:main`); `start_proxy.sh` is compatibility-only.
- Anthropic header validation is configurable (`ANTHROPIC_SUPPORTED_VERSIONS`, fallback/missing flags), defaulting to `2023-06-01`.
- Contract API routes include both `request-id` and `x-request-id` headers on success and error paths (`/v1/messages*`, `/v1/models*`, and removed OpenAI generation endpoints).
- Integration coverage is script-driven (`python tests/test_main.py`); pytest config exists but is not the primary repo flow.
- CI quality gates run `isort`, `black`, `ruff`, `mypy`, and `compileall`; CI does not run live integration scripts.
- Docker image context is intentionally minimal (`app/**` + `pyproject.toml`) via whitelist `.dockerignore`.

## ANTI-PATTERNS (THIS PROJECT)

- Do not rely on stale docs command `python app/test_claude_to_openai.py` (file does not exist).
- Do not use removed generation routes `POST /v1/chat/completions` or `POST /v1/responses`; they intentionally return 404.
- Do not assume `docker-compose.yml` exists; this repo currently has no compose file.
- Do not diverge CI and Docker install/run workflows from the `pip` commands documented in this guide.
- Do not remove request-id header parity (`request-id` and `x-request-id`) from API responses.
- Do not treat provider-dependent integration skips/timeouts as deterministic product regressions.

## UNIQUE STYLES

- One router serves Anthropic-compatible generation and OpenAI-compatible model discovery with explicit contract separation.
- Conversion layer handles both Responses-style events and legacy chat-completion chunk shapes while preserving Claude SSE semantics.
- Cancellation is coordinated across API, conversion, and client layers via shared `request_id`.
- Optional Anthropic compatibility metadata can be attached into provider `extra_body.proxy_metadata` when enabled.

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
python tests/test_main.py
docker build -t clay .
```

## NOTES

- `tests/test_main.py` is the canonical integration harness and executes sequentially via `asyncio.run(main())`.
- `.github/workflows/docker-images.yml` builds multi-arch images (`linux/amd64,linux/arm64`) and only pushes outside pull requests.
- `app/models/openai.py` defines compatibility request models; active OpenAI-compatible HTTP routes currently expose only `/v1/models*`.
- `config = Config()` runs at import time and exits process on invalid required env or invalid state mode.
- `README.md` provides operator quickstart; AGENTS guides remain the detailed in-repo implementation reference.
