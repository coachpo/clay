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
- `src/**` -> `src/AGENTS.md`
- `src/api/**` -> `src/api/AGENTS.md`
- `src/core/**` -> `src/core/AGENTS.md`
- `src/conversion/**` -> `src/conversion/AGENTS.md`
- `src/models/**` -> `src/models/AGENTS.md`
- `tests/**` and `test_cancellation.py` -> `tests/AGENTS.md`

## STRUCTURE
```text
clay/
|-- src/
|   |-- main.py                  # FastAPI app + exception adapters + uvicorn launcher
|   |-- api/endpoints.py         # Anthropic/OpenAI routes, validation gates, cancellation wiring
|   |-- core/                    # config, OpenAI client, logging, constants, model mapping
|   |-- conversion/              # Claude<->OpenAI request/response + SSE conversion
|   `-- models/                  # strict Claude schema + permissive OpenAI compatibility models
|-- tests/test_main.py           # async integration scenario runner
|-- test_cancellation.py         # cancellation/disconnect scenario runner
|-- .github/workflows/           # CI quality, Docker image build, cleanup automation
|-- start_proxy.py               # compatibility wrapper entry script (mutates sys.path)
|-- pyproject.toml               # packaging entrypoint + tool config
`-- Dockerfile                   # production container build
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Anthropic request flow | `src/api/endpoints.py` | `/v1/messages` validation, conversion, stream/non-stream paths |
| Token counting behavior | `src/api/endpoints.py` | `/v1/messages/count_tokens` estimation rules |
| OpenAI-compatible surface | `src/api/endpoints.py` | `/v1/models*` only; generation routes intentionally return 404 |
| Request conversion | `src/conversion/request_converter.py` | Claude blocks/tools/thinking -> OpenAI Responses payload |
| Stream conversion | `src/conversion/response_converter.py` | OpenAI SSE/chunk events -> Claude SSE event order |
| Provider transport + cancellation | `src/core/client.py` | async OpenAI calls + `request_id` cancellation map |
| Config/env behavior | `src/core/config.py` | import-time validation, compatibility flags, state-mode guard |
| Schema constraints | `src/models/claude.py`, `src/models/openai.py` | strict Claude validation vs permissive OpenAI model shapes |
| Integration behavior checks | `tests/test_main.py`, `test_cancellation.py` | script-style end-to-end checks and cancellation probes |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `create_message` | async function | `src/api/endpoints.py` | high | Main Anthropic request lifecycle |
| `_create_response_with_disconnect_cancellation` | async function | `src/api/endpoints.py` | medium | Non-stream disconnect cancellation |
| `OpenAIClient.create_response_stream` | async method | `src/core/client.py` | high | Streaming Responses provider transport |
| `convert_claude_to_openai` | function | `src/conversion/request_converter.py` | high | Claude -> OpenAI payload translation |
| `convert_openai_streaming_to_claude_with_cancellation` | async function | `src/conversion/response_converter.py` | high | OpenAI stream -> Claude SSE bridge |
| `ClaudeMessagesRequest` | class | `src/models/claude.py` | medium | Strict Claude request schema + validators |
| `ModelManager.map_claude_model_to_openai` | method | `src/core/model_manager.py` | medium | Model routing policy |

## CONVENTIONS
- Use `pyproject.toml` as the dependency source of truth (`python -m pip install .[dev]` for local quality checks).
- Canonical runtime entrypoint is `clay` (`pyproject` script -> `src.main:main`); `start_proxy.py` is compatibility-only.
- Anthropic header validation is configurable (`ANTHROPIC_SUPPORTED_VERSIONS`, fallback/missing flags), defaulting to `2023-06-01`.
- Responses include both `request-id` and `x-request-id` headers on success and error paths.
- Integration coverage is script-driven (`python tests/test_main.py`, `python test_cancellation.py`); pytest config exists in `pyproject.toml` but is not the primary flow.
- CI quality gates run `isort`, `black`, `ruff`, `mypy`, and `compileall`; CI does not run live integration scripts.

## ANTI-PATTERNS (THIS PROJECT)
- Do not rely on stale docs command `python src/test_claude_to_openai.py` (file does not exist).
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
python start_proxy.py
isort --check-only src tests test_cancellation.py
black --check src tests test_cancellation.py
ruff check src tests test_cancellation.py
mypy src
python tests/test_main.py
python test_cancellation.py
docker build -t clay .
```

## NOTES
- `.dockerignore` excludes `AGENTS.md`, `tests/`, and `test_cancellation.py` from image build context.
- `src/models/openai.py` defines compatibility request models; active HTTP routes currently expose only `/v1/models*` on the OpenAI-compatible surface.
- `config = Config()` runs at import time and exits process on invalid required env or invalid state mode.
