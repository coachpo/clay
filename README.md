# Clay

Clay is a FastAPI proxy that uses LiteLLM to translate between Anthropic Messages API format and OpenAI-compatible providers.

## Features

- **Anthropic Messages API compatibility**: Accept requests in Anthropic format
- **LiteLLM-powered routing**: Uses LiteLLM for provider access and OpenAI-compatible responses
- **Thin compatibility adapters**: Converts Anthropic requests and responses at the proxy edge
- **Streaming support**: Full SSE streaming for real-time responses
- **Model mapping**: Flexible Claude model → OpenAI model mapping
- **Simple configuration**: Minimal environment variables required

## Architecture

Clay v2.0 is a complete rewrite using LiteLLM Python SDK:

- **LiteLLM handles transport**: Provider access, retries, and OpenAI-compatible responses
- **Clay handles Anthropic compatibility**: Lightweight request/response conversion around LiteLLM
- **Simplified codebase**: Substantially reduced complexity from v1.x
- **Production-ready**: Built on battle-tested LiteLLM library

## Requirements

- Python `>=3.13`
- `OPENAI_API_KEY` environment variable

## Quick Start

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

Optional:
- `ANTHROPIC_API_KEY` (enforce client key validation)
- `BIG_MODEL`, `MIDDLE_MODEL`, `SMALL_MODEL` (model mapping)

3. Run:

```bash
clay
```

Or:

```bash
python -m app.main
```

## API Endpoints

### Anthropic Messages API

```bash
POST /v1/messages
```

Accepts Anthropic Messages API format, maps the Claude model to a configured OpenAI-compatible model, and forwards the request through LiteLLM.

Example:

```bash
curl http://localhost:8000/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: your-anthropic-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Utility Endpoints

- `GET /health` - Health check
- `GET /` - Service info

## Environment Variables

### Required

- `OPENAI_API_KEY` - Your OpenAI API key (used by LiteLLM)

### Optional

- `ANTHROPIC_API_KEY` - Expected client API key for validation
- `BIG_MODEL` - Model for opus requests (default: `gpt-4o`)
- `MIDDLE_MODEL` - Model for sonnet requests (default: `gpt-4o`)
- `SMALL_MODEL` - Model for haiku requests (default: `gpt-4o-mini`)
- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8000`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `UVICORN_WORKERS` - Worker process count (default: `1`)
- `MAX_TOKENS_LIMIT` - Token limit (default: `4096`)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `90`)

## Model Mapping

Clay maps Claude model names to OpenAI models:

- Models containing `haiku` → `SMALL_MODEL`
- Models containing `sonnet` → `MIDDLE_MODEL`
- Models containing `opus` → `BIG_MODEL`
- Pass-through prefixes: `gpt-`, `o1-`, `o3-`, `o4-`, `gpt-5`, `ep-`, `doubao-`, `deepseek-`

## Testing

Run contract tests:

```bash
pytest tests/test_main.py
```

Quality checks:

```bash
isort --check-only app tests
black --check app tests
ruff check app tests
mypy app
```

## Migration from v1.x

Clay v2.0 is a breaking rewrite:

- **Removed**: OpenAI Responses API support
- **Removed**: Heavy custom client/retry stack
- **Simplified**: Anthropic compatibility layer
- **Simplified**: Configuration (fewer environment variables)
- **Simplified**: Codebase footprint relative to v1.x

## How It Works

1. Client sends Anthropic Messages API request
2. Clay validates API key and headers
3. Clay converts the request into OpenAI-compatible chat completion parameters
4. LiteLLM forwards the request to the mapped provider model
5. Clay converts the LiteLLM response back to Anthropic format
6. Clay returns Anthropic-compatible JSON or SSE events

## License

MIT
