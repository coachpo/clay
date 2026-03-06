# APP KNOWLEDGE BASE

## READ WHEN
- Working in any `app/**` module.
- A change spans API routes, conversion logic, client behavior, and schema validation together.

## CHILD GUIDES
- `app/api/**` -> `app/api/AGENTS.md`
- `app/core/**` -> `app/core/AGENTS.md`
- `app/conversion/**` -> `app/conversion/AGENTS.md`
- `app/models/**` -> `app/models/AGENTS.md`
- Nearest child guide overrides this file for local details.

## APP-SCOPE RESPONSIBILITIES
- `main.py`: app construction, path-aware exception envelope shaping, and CLI startup (`clay`).
- `api/endpoints.py`: route contract enforcement and request lifecycle orchestration.
- `core/*`: runtime config, provider transport/retries/cancellation, model routing, constants, shared logging.
- `conversion/*`: Claude <-> OpenAI payload/event translation.
- `models/*`: strict Claude request schemas and OpenAI compatibility request models.

## CROSS-MODULE INVARIANTS
- Keep route-family error envelope behavior path-aware (`/v1/messages*` Anthropic shape, `/v1/models*` OpenAI shape).
- Keep request-id threading consistent through API -> conversion -> client cancellation flow.
- Keep sampling compatibility semantics uniform (`temperature`/`top_p` accepted, ignored upstream).
- Keep fallback retry semantics shared across stream and non-stream paths (`metadata`, `context_management`, `extra_body`).
- Keep removed generation compatibility routes returning `404` in API surface.

## CHANGE GUIDELINES
- Prefer narrow fixes in `app/api/endpoints.py`; helper behavior is reused by multiple routes.
- If changing schema fields in `app/models`, update converters and endpoint assumptions in the same change.
- If changing converter behavior, verify stream and non-stream paths remain contract-consistent.
- If changing startup/logging behavior, keep `app/main.py` and `app/core/logging.py` semantics aligned.

## VERIFICATION
```bash
mypy app
python tests/test_main.py
```
