# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-01T22:18:01+0200  
**Commit:** af8e3ec  
**Branch:** main

## OVERVIEW
Clay exposes Claude-compatible and OpenAI-compatible chat endpoints and forwards requests to OpenAI-compatible providers.
Runtime is Python/FastAPI with explicit request validation, model remapping, cancellation-aware transport, and SSE protocol conversion.

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
|   |-- main.py                # FastAPI app + exception adapters + uvicorn launcher
|   |-- api/endpoints.py       # Anthropic/OpenAI routes + validation + cancellation wiring
|   |-- core/                  # config, OpenAI client, logging, constants, model mapping
|   |-- conversion/            # Claude<->OpenAI request/response + SSE conversion
|   `-- models/                # strict Claude schema + permissive OpenAI request schema
|-- tests/test_main.py         # async integration scenario runner
|-- test_cancellation.py       # cancellation/disconnect scenario runner
|-- start_proxy.py             # wrapper entry script (mutates sys.path)
|-- pyproject.toml             # packaging entrypoint + tool config
|-- Dockerfile                 # lockfile-enforced container build
`-- docker-compose.yml         # local container wiring
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Anthropic request flow | `src/api/endpoints.py` | `/v1/messages` validation, conversion, streaming/non-stream paths |
| OpenAI-compatible routes | `src/api/endpoints.py` | `/v1/chat/completions`, `/v1/responses`, `/v1/models` |
| Request conversion | `src/conversion/request_converter.py` | Claude blocks/tools/thinking -> OpenAI payload |
| Stream conversion | `src/conversion/response_converter.py` | SSE event order + tool-call delta reconstruction |
| Provider transport + cancellation | `src/core/client.py` | async OpenAI calls + `request_id` cancellation map |
| Config and env behavior | `src/core/config.py` | import-time validation and runtime defaults |
| Model name policy | `src/core/model_manager.py` | pass-through and Claude family mapping |
| Schema constraints | `src/models/claude.py`, `src/models/openai.py` | strict Claude validation vs permissive OpenAI request models |
| Integration behavior checks | `tests/test_main.py`, `test_cancellation.py` | script-style end-to-end scenarios |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `create_message` | async function | `src/api/endpoints.py` | high | Main Anthropic request lifecycle |
| `_create_chat_completion_with_disconnect_cancellation` | async function | `src/api/endpoints.py` | medium | Non-stream disconnect cancellation |
| `OpenAIClient.create_chat_completion_stream` | async method | `src/core/client.py` | high | Streaming provider transport |
| `convert_claude_to_openai` | function | `src/conversion/request_converter.py` | high | Claude->OpenAI payload translation |
| `convert_openai_streaming_to_claude_with_cancellation` | async function | `src/conversion/response_converter.py` | high | OpenAI SSE->Claude SSE bridge |
| `ClaudeMessagesRequest` | class | `src/models/claude.py` | medium | Strict Claude request schema + validators |
| `ModelManager.map_claude_model_to_openai` | method | `src/core/model_manager.py` | medium | Model routing policy |

## CONVENTIONS
- Use `uv` workflow first (`uv sync`, `uv run ...`); `pip` via `requirements.txt` is fallback.
- Docker build expects lockfile correctness (`uv sync --locked` in `Dockerfile`).
- `clay` entrypoint resolves to `src.main:main`; `start_proxy.py` is a compatibility wrapper.
- Anthropic contract enforces `anthropic-version: 2023-06-01` and JSON content type.
- Responses include both `request-id` and `x-request-id` headers.
- Integration tests are executable async scripts; no repo-level pytest config.

## ANTI-PATTERNS (THIS PROJECT)
- Do not rely on stale docs command `python src/test_claude_to_openai.py` (file does not exist).
- Do not bypass lockfile sync in container builds (`uv sync --locked` is required path).
- Do not assume `start_proxy.py` is the canonical runtime path; prefer packaged entrypoint for stable imports.
- Do not treat script-style tests as isolated unit tests; many scenarios hit live provider routes.
- Do not assume `src/models/openai.py` is empty; endpoint validation now depends on its request models.

## UNIQUE STYLES
- One router serves Anthropic-compatible and OpenAI-compatible surfaces from the same module.
- Conversion layer preserves tool-call identity and Claude SSE ordering semantics.
- Cancellation is coordinated across API, conversion, and client layers via shared `request_id`.
- Model mapping supports direct pass-through for `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `gpt-5`, `ep-*`, `doubao-*`, `deepseek-*`.

## COMMANDS
```bash
uv sync
uv run clay
python start_proxy.py
docker compose up -d
uv run black src/
uv run isort src/
uv run mypy src/
python tests/test_main.py
python test_cancellation.py
```

## NOTES
- `src/core/config.py` initializes `config` at import time and exits process on invalid required env.
- LSP `documentSymbol` support is unavailable in this environment; symbol map used AST-based extraction.
