# PROJECT KNOWLEDGE BASE

## OVERVIEW
Clay v2.0 is a FastAPI proxy using LiteLLM Python SDK to translate between Anthropic Messages API format and OpenAI-compatible providers.

## ARCHITECTURE
- **LiteLLM-powered transport**: LiteLLM handles provider access, retries, and OpenAI-compatible responses
- **Thin compatibility layer**: Clay keeps lightweight Anthropic request/response conversion around LiteLLM
- **Simplified**: Significantly reduced surface area from v1.x
- **Single endpoint**: POST /v1/messages (Anthropic Messages API)

## STRUCTURE MAP
```text
clay/
|-- app/
|   |-- main.py                  # FastAPI app + exception handlers + CLI
|   |-- api/endpoints.py         # Single /v1/messages endpoint using litellm.acompletion
|   |-- conversion/
|   |   |-- request_converter.py # Anthropic request -> OpenAI chat completion params
|   |   `-- response_converter.py# OpenAI/LiteLLM output -> Anthropic response/SSE
|   |-- core/
|   |   |-- config.py            # Environment config (simplified)
|   |   |-- constants.py         # Shared Anthropic/OpenAI conversion constants
|   |   `-- model_mapper.py      # Claude model name → OpenAI model name mapping
|   `-- models/
|       `-- claude.py            # Lightweight Anthropic request models
|-- tests/test_main.py           # Local API contract tests with mocked LiteLLM
|-- pyproject.toml               # Dependencies: litellm>=1.80.0
`-- README.md                    # User documentation
```

## CORE CONTRACTS
- Accept Anthropic Messages API format at POST /v1/messages
- Use `litellm.acompletion(model=<mapped_openai_model>, ...)` through LiteLLM
- Return Anthropic-compatible responses via Clay's compatibility converters
- Preserve `request-id` and `x-request-id` headers
- Support streaming via SSE

## KEY BEHAVIOR
- **Model mapping**: Claude model names → OpenAI models via ModelMapper
- **API key validation**: Optional ANTHROPIC_API_KEY for client validation
- **Error handling**: LiteLLM exceptions → Anthropic error format
- **Streaming**: FastAPI StreamingResponse + litellm.acompletion(stream=True)
- **Disconnect handling**: Check request.is_disconnected() in stream loop

## REMOVED FROM V1.X
- app/core/client.py (OpenAIClient with retry logic)
- app/core/logging.py, model_manager.py
- app/models/openai.py, complex Claude schemas
- OpenAI Responses API support
- Anthropic version negotiation
- Context management, reasoning effort mapping

## LITELLM INTEGRATION
- `litellm.acompletion()` for async requests
- Mapped OpenAI-compatible model ids via `ModelMapper`
- `stream=True` for streaming responses
- `litellm.drop_params = True` to ignore unsupported params
- Exception types: AuthenticationError, RateLimitError, ContextWindowExceededError, Timeout, BadRequestError

## TESTING
```bash
pytest tests/test_main.py  # Contract tests
mypy app                   # Type checking
```

## ANTI-PATTERNS
- Do not bypass LiteLLM for Anthropic requests
- Do not reintroduce heavyweight request/response machinery from v1.x
- Do not implement custom retry logic (LiteLLM has built-in retries)

## CHANGE CHECKLIST
- Update endpoints.py for API changes
- Update config.py for new environment variables
- Update tests/test_main.py for behavior changes
- Update README.md for user-facing changes
