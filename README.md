# Clay

Clay is a FastAPI proxy that exposes Anthropic-style generation endpoints and forwards them to OpenAI-compatible Responses providers.

It provides:
- Anthropic-compatible generation routes:
  - `POST /v1/messages`
  - `POST /v1/messages/count_tokens`
- OpenAI-compatible model discovery routes:
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`

## Key behavior

- Request/response contract bridging between Anthropic Messages and OpenAI Responses.
- Claude-style SSE event shape/order for streaming responses.
- Request-scoped cancellation support.
- `request-id` and `x-request-id` parity headers on success and errors.
- Optional retry fallback that strips optional fields (`metadata`, `context_management`, `extra_body`) when upstreams reject them or return retryable/protocol failures.
- Sampling policy: `temperature` and `top_p` are accepted for Anthropic compatibility but always ignored (not forwarded upstream).

## Requirements

- Python `>=3.13`
- A valid `OPENAI_API_KEY`

## Quick start

1. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

2. Configure environment:

```bash
cp .env.example .env
```

Set at least:
- `OPENAI_API_KEY`

Optional but common:
- `ANTHROPIC_API_KEY` (enforce client key validation)
- `OPENAI_BASE_URL` (for OpenAI-compatible upstreams)
- `BIG_MODEL`, `MIDDLE_MODEL`, `SMALL_MODEL`
- `OPENAI_RESPONSES_STATE_MODE` (`stateless` or `provider`)

3. Run:

```bash
clay
```

Or:

```bash
python -m app.main
```

## Minimal request example

```bash
curl http://localhost:8000/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: your-expected-anthropic-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-opus-4-6",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Environment variables

Required:
- `OPENAI_API_KEY`

Core runtime:
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `AZURE_API_VERSION` (optional, for Azure-style base URLs)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `LOG_LEVEL` (default: `INFO`)
- `UVICORN_WORKERS` (default: `1`)
- `REQUEST_TIMEOUT` (default: `90`)
- `MAX_RETRIES` (default: `2`)

Model routing:
- `BIG_MODEL`
- `MIDDLE_MODEL` (defaults to `BIG_MODEL` when unset)
- `SMALL_MODEL`

Anthropic contract controls:
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_DEFAULT_VERSION` (default: `2023-06-01`)
- `ANTHROPIC_SUPPORTED_VERSIONS` (CSV list)
- `ANTHROPIC_ALLOW_VERSION_FALLBACK`
- `ANTHROPIC_ALLOW_MISSING_VERSION`
- `ANTHROPIC_ALLOW_UNKNOWN_FIELDS`
- `ANTHROPIC_COMPATIBILITY_MODE`

Compatibility metadata passthrough:
- `ALLOW_OPENAI_EXTENSION_PASSTHROUGH`
- `INCLUDE_ORIGINAL_ANTHROPIC_REQUEST`

State mode:
- `OPENAI_RESPONSES_STATE_MODE`:
  - `stateless` (default)
  - `provider`

## Quality checks

```bash
isort --check-only app tests
black --check app tests
ruff check app tests
mypy app
```

Integration harness:

```bash
python tests/test_main.py
```

## Notes

- `POST /v1/chat/completions` and `POST /v1/responses` are intentionally removed from proxy surface and return 404.
- This repo's canonical behavior checks are centered in `tests/test_main.py`.
