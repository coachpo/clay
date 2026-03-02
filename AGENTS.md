# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-02T17:28:29+02:00  
**Commit:** 0a44782  
**Branch:** main

## OVERVIEW

Clay exposes Anthropic-compatible generation (`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI-compatible model discovery (`/v1/models*`).
Runtime is Python/FastAPI; generation is routed upstream through OpenAI Responses API with explicit conversion, SSE bridging, and cancellation wiring.

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
|   |-- main.py                  # FastAPI app + exception adapters + uvicorn launcher
|   |-- api/endpoints.py         # Anthropic/OpenAI routes, validation gates, cancellation wiring
|   |-- core/                    # config, OpenAI client, logging, constants, model mapping
|   |-- conversion/              # Claude<->OpenAI request/response + SSE conversion
|   `-- models/                  # strict Claude schema + permissive OpenAI compatibility models
|-- tests/test_main.py           # async integration scenario runner
|-- .github/workflows/           # CI quality, Docker image build, cleanup automation
|-- start_proxy.sh               # compatibility wrapper shell script
|-- pyproject.toml               # packaging entrypoint + tool config
`-- Dockerfile                   # production container build
```

## WHERE TO LOOK

| Task                              | Location                                       | Notes                                                          |
| --------------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| Anthropic request flow            | `app/api/endpoints.py`                         | `/v1/messages` validation, conversion, stream/non-stream paths |
| Token counting behavior           | `app/api/endpoints.py`                         | `/v1/messages/count_tokens` estimation rules                   |
| OpenAI-compatible surface         | `app/api/endpoints.py`                         | `/v1/models*` only; generation routes intentionally return 404 |
| Request conversion                | `app/conversion/request_converter.py`          | Claude blocks/tools/thinking -> OpenAI Responses payload       |
| Stream conversion                 | `app/conversion/response_converter.py`         | OpenAI SSE/chunk events -> Claude SSE event order              |
| Provider transport + cancellation | `app/core/client.py`                           | async OpenAI calls + `request_id` cancellation map             |
| Config/env behavior               | `app/core/config.py`                           | import-time validation, compatibility flags, state-mode guard  |
| Schema constraints                | `app/models/claude.py`, `app/models/openai.py` | strict Claude validation vs permissive OpenAI model shapes     |
| Integration behavior checks       | `tests/test_main.py`                           | script-style end-to-end checks                                 |

## CODE MAP

| Symbol                                                 | Type           | Location                               | Refs   | Role                                      |
| ------------------------------------------------------ | -------------- | -------------------------------------- | ------ | ----------------------------------------- |
| `create_message`                                       | async function | `app/api/endpoints.py`                 | high   | Main Anthropic request lifecycle          |
| `_create_response_with_disconnect_cancellation`        | async function | `app/api/endpoints.py`                 | medium | Non-stream disconnect cancellation        |
| `OpenAIClient.create_response_stream`                  | async method   | `app/core/client.py`                   | high   | Streaming Responses provider transport    |
| `convert_claude_to_openai`                             | function       | `app/conversion/request_converter.py`  | high   | Claude -> OpenAI payload translation      |
| `convert_openai_streaming_to_claude_with_cancellation` | async function | `app/conversion/response_converter.py` | high   | OpenAI stream -> Claude SSE bridge        |
| `ClaudeMessagesRequest`                                | class          | `app/models/claude.py`                 | medium | Strict Claude request schema + validators |
| `ModelManager.map_claude_model_to_openai`              | method         | `app/core/model_manager.py`            | medium | Model routing policy                      |

## CONVENTIONS

- Use `pyproject.toml` as the dependency source of truth (`python -m pip install .[dev]` for local quality checks).
- Canonical runtime entrypoint is `clay` (`pyproject` script -> `app.main:main`); `start_proxy.sh` is compatibility-only.
- Anthropic header validation is configurable (`ANTHROPIC_SUPPORTED_VERSIONS`, fallback/missing flags), defaulting to `2023-06-01`.
- Responses include both `request-id` and `x-request-id` headers on success and error paths.
- Integration coverage is script-driven (`python tests/test_main.py`); pytest config exists in `pyproject.toml` but is not the primary flow.
- CI quality gates run `isort`, `black`, `ruff`, `mypy`, and `compileall`; CI does not run live integration scripts.

## ANTI-PATTERNS (THIS PROJECT)

- Do not rely on stale docs command `python app/test_claude_to_openai.py` (file does not exist).
- Do not use removed generation routes `POST /v1/chat/completions` or `POST /v1/responses`; they intentionally return 404.
- Do not assume `docker-compose.yml` exists; this repo currently has no compose file.
- Do not diverge CI and Docker install/run workflows from the `pip` commands documented in this guide.
- Do not remove request-id header parity (`request-id` and `x-request-id`) from API responses.
- Do not treat provider-dependent integration skips/timeouts as deterministic product regressions.

## UNIQUE STYLES

- One router serves Anthropic-compatible generation and OpenAI-compatible model discovery with explicit contract separation.
- Conversion layer supports both Responses-style events and legacy chunk shapes while preserving Claude SSE semantics.
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

- `.dockerignore` excludes `AGENTS.md` and `tests/` from image build context.
- `app/models/openai.py` defines compatibility request models; active HTTP routes currently expose only `/v1/models*` on the OpenAI-compatible surface.
- `config = Config()` runs at import time and exits process on invalid required env or invalid state mode.
