import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import (
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async OpenAI client with cancellation support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 90,
        max_retries: int = 2,
        api_version: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        if api_version:
            self.base_url = base_url.strip()
        else:
            self.base_url = self._normalize_base_url(base_url)

        self.client: Any
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.base_url,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        self.active_requests: Dict[str, asyncio.Event] = {}

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        raw = base_url.strip()
        if not raw:
            return raw

        parsed = urlsplit(raw)
        path = parsed.path.rstrip("/")
        if path in {"", "/"}:
            path = "/v1"

        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    @staticmethod
    def _is_unsupported_parameter_error(error: BadRequestError, parameter: str) -> bool:
        detail = str(error).lower()
        normalized_parameter = parameter.lower()
        if normalized_parameter not in detail:
            return False
        return "unsupported parameter" in detail or "unknown parameter" in detail

    @staticmethod
    def _is_unexpected_keyword_argument_error(error: TypeError, parameter: str) -> bool:
        detail = str(error).lower()
        normalized_parameter = parameter.lower()
        if normalized_parameter not in detail:
            return False
        return "unexpected keyword argument" in detail

    def _remove_optional_fallback_fields(
        self,
        request: Dict[str, Any],
        fallback_fields: tuple[str, ...],
        error: Exception,
    ) -> tuple[Dict[str, Any], tuple[str, ...], Optional[str]]:
        trigger_field: Optional[str] = None
        for field in fallback_fields:
            if field not in request:
                continue

            if isinstance(error, BadRequestError):
                is_unsupported = self._is_unsupported_parameter_error(error, field)
            elif isinstance(error, TypeError):
                is_unsupported = self._is_unexpected_keyword_argument_error(error, field)
            else:
                is_unsupported = False

            if is_unsupported:
                trigger_field = field
                break

        if trigger_field is None:
            return request, (), None

        next_request, removed_fields = self._remove_present_optional_fields(
            request, fallback_fields
        )

        return next_request, removed_fields, trigger_field

    @staticmethod
    def _remove_present_optional_fields(
        request: Dict[str, Any],
        fallback_fields: tuple[str, ...],
    ) -> tuple[Dict[str, Any], tuple[str, ...]]:
        next_request = dict(request)
        removed_fields: list[str] = []
        for field in fallback_fields:
            if field in next_request:
                next_request.pop(field, None)
                removed_fields.append(field)
        return next_request, tuple(removed_fields)

    @staticmethod
    def _is_retryable_server_status(error: APIError) -> bool:
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and status_code in {
            500,
            502,
            503,
            504,
            520,
            521,
            522,
            523,
            524,
            525,
            526,
            527,
            529,
        }

    @staticmethod
    def _normalize_error_status_code(status_code: Any) -> int:
        if isinstance(status_code, int) and status_code >= 400:
            return status_code
        return 502

    @staticmethod
    def _content_type_is_json(content_type: str) -> bool:
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type == "application/json" or media_type.endswith("+json")

    @staticmethod
    def _looks_like_html_error_payload(payload_preview: str) -> bool:
        lowered = payload_preview.lower()
        markers = (
            "<!doctype html",
            "<html",
            "cloudflare",
            "cf-error",
            "error 1020",
            "ray id",
            "bad gateway",
            "access denied",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _safe_payload_preview(payload: Any, max_chars: int = 240) -> str:
        if payload is None:
            return ""

        if isinstance(payload, (dict, list)):
            raw = json.dumps(payload, ensure_ascii=False)
        else:
            raw = str(payload)

        normalized = " ".join(raw.split())
        return normalized[:max_chars]

    def _build_upstream_protocol_error(self, error: APIError) -> HTTPException:
        raw_status_code = getattr(error, "status_code", None)
        status_code = self._normalize_error_status_code(raw_status_code)
        response = getattr(error, "response", None)
        content_type = ""
        cf_mitigated = ""
        cf_ray = ""

        body_preview = self._safe_payload_preview(getattr(error, "body", None))
        if response is not None:
            try:
                content_type = str(response.headers.get("content-type", ""))
                cf_mitigated = str(response.headers.get("cf-mitigated", ""))
                cf_ray = str(response.headers.get("cf-ray", ""))
            except Exception:
                content_type = ""
                cf_mitigated = ""
                cf_ray = ""

            if not body_preview:
                try:
                    body_preview = self._safe_payload_preview(response.text)
                except Exception:
                    body_preview = ""

        hints: list[str] = []
        if content_type and not self._content_type_is_json(content_type):
            hints.append(f"unexpected content-type '{content_type}'")
        if cf_mitigated:
            hints.append(f"cf-mitigated={cf_mitigated}")
        if cf_ray:
            hints.append(f"cf-ray={cf_ray}")
        if body_preview and self._looks_like_html_error_payload(body_preview):
            hints.append("html error payload signature")

        hint_suffix = f" ({', '.join(hints)})" if hints else ""
        message = "Upstream protocol error: expected JSON API response payload"
        if body_preview:
            message = f"{message}{hint_suffix}. Body preview: {body_preview}"
        else:
            message = f"{message}{hint_suffix}."

        return HTTPException(status_code=status_code, detail=message)

    @staticmethod
    def _is_likely_json_parse_error(detail: str) -> bool:
        lowered = detail.lower()
        markers = (
            "expecting value",
            "invalid character",
            "bad_response_body",
            "beginning of value",
            "response body is not valid json",
            "error parsing response body",
            "failed to parse response body",
            "unable to decode response body",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _extract_api_error_block(error: APIError) -> Dict[str, Any]:
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            return {}
        error_block = body.get("error")
        if isinstance(error_block, dict):
            return error_block
        return {}

    def _is_protocol_error_api_error(self, error: APIError) -> bool:
        raw_status_code = getattr(error, "status_code", None)
        if isinstance(error, APIResponseValidationError) or not (
            isinstance(raw_status_code, int) and raw_status_code >= 400
        ):
            return True

        response = getattr(error, "response", None)
        if response is not None:
            try:
                content_type = str(response.headers.get("content-type", ""))
            except Exception:
                content_type = ""
            if content_type and not self._content_type_is_json(content_type):
                return True

        error_block = self._extract_api_error_block(error)
        error_code = str(error_block.get("code", "")).lower()
        error_type = str(error_block.get("type", "")).lower()
        error_message = str(error_block.get("message", "")).lower()
        protocol_codes = {
            "bad_response_body",
            "invalid_json",
            "json_parse_error",
            "response_validation_error",
        }
        if error_code in protocol_codes or error_type in protocol_codes:
            return True
        if error_message and self._is_likely_json_parse_error(error_message):
            return True

        body_preview = self._safe_payload_preview(getattr(error, "body", None))
        if body_preview and self._looks_like_html_error_payload(body_preview):
            return True

        return self._is_likely_json_parse_error(str(error))

    def _build_protocol_error_from_exception(self, error: Exception) -> HTTPException:
        detail_preview = self._safe_payload_preview(str(error), max_chars=160)
        message = "Upstream protocol error: expected JSON API response payload"
        if detail_preview:
            message = f"{message}. Parse failure: {detail_preview}"
        else:
            message = f"{message}."
        return HTTPException(status_code=502, detail=message)

    def _is_protocol_error_exception(self, error: Exception) -> bool:
        if isinstance(error, json.JSONDecodeError):
            return True

        if isinstance(error, ValueError) and self._is_likely_json_parse_error(str(error)):
            return True

        return False

    async def _create_with_metadata_fallback(self, request: Dict[str, Any]) -> Any:
        # Some OpenAI-compatible providers reject selected optional request fields.
        # Retry once without all known compatibility fields when any one is rejected.
        retry_request = dict(request)
        # These are compatibility-only fields; if upstream rejects them or emits transient gateway
        # failures, retry once without them to maximize request success.
        fallback_fields = ("metadata", "context_management", "extra_body")

        try:
            return await self.client.responses.create(**retry_request)
        except (BadRequestError, TypeError) as error:
            retry_request, removed_fields, trigger_field = self._remove_optional_fallback_fields(
                retry_request,
                fallback_fields,
                error,
            )
            if trigger_field is None:
                raise

            logger.warning(
                "Upstream rejected optional field '%s' (%s); retrying /v1/responses without optional fields: %s.",
                trigger_field,
                type(error).__name__,
                ", ".join(removed_fields),
            )
            return await self.client.responses.create(**retry_request)
        except APIError as error:
            # Some gateways return transient 5xx (e.g. 502) for optional fields like context_management.
            # Retry once without optional compatibility fields before surfacing the upstream failure.
            if self._is_retryable_server_status(error):
                retry_request, removed_fields = self._remove_present_optional_fields(
                    retry_request, fallback_fields
                )
                if not removed_fields:
                    raise

                logger.warning(
                    "Upstream returned %s; retrying /v1/responses without optional fields: %s.",
                    getattr(error, "status_code", "unknown"),
                    ", ".join(removed_fields),
                )
                return await self.client.responses.create(**retry_request)

            if self._is_protocol_error_api_error(error):
                retry_request, removed_fields = self._remove_present_optional_fields(
                    retry_request, fallback_fields
                )
                if removed_fields:
                    logger.warning(
                        "Upstream protocol parse error (%s); retrying /v1/responses without optional fields: %s.",
                        type(error).__name__,
                        ", ".join(removed_fields),
                    )
                else:
                    logger.warning(
                        "Upstream protocol parse error (%s); retrying /v1/responses once.",
                        type(error).__name__,
                    )
                return await self.client.responses.create(**retry_request)

            raise
        except (json.JSONDecodeError, ValueError) as error:
            if not self._is_likely_json_parse_error(str(error)):
                raise

            retry_request, removed_fields = self._remove_present_optional_fields(
                retry_request, fallback_fields
            )
            if removed_fields:
                logger.warning(
                    "Upstream JSON parse failure (%s); retrying /v1/responses without optional fields: %s.",
                    type(error).__name__,
                    ", ".join(removed_fields),
                )
            else:
                logger.warning(
                    "Upstream JSON parse failure (%s); retrying /v1/responses once.",
                    type(error).__name__,
                )
            return await self.client.responses.create(**retry_request)

    async def create_response(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send non-streaming Responses API request with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            response_task = asyncio.create_task(self._create_with_metadata_fallback(request))

            if request_id:
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [response_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for pending_task in pending:
                    pending_task.cancel()
                    try:
                        await pending_task
                    except asyncio.CancelledError:
                        pass

                if cancel_task in done:
                    response_task.cancel()
                    raise HTTPException(
                        status_code=499,
                        detail="Request cancelled by client",
                    )

                response = await response_task
            else:
                response = await response_task

            return dict(response.model_dump())

        except HTTPException:
            raise
        except AuthenticationError as error:
            raise HTTPException(
                status_code=401,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except RateLimitError as error:
            raise HTTPException(
                status_code=429,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except BadRequestError as error:
            raise HTTPException(
                status_code=400,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except APIError as error:
            if self._is_protocol_error_api_error(error):
                raise self._build_upstream_protocol_error(error) from error

            raw_status_code = getattr(error, "status_code", None)
            raise HTTPException(
                status_code=self._normalize_error_status_code(raw_status_code),
                detail=self.classify_openai_error(str(error)),
            ) from error
        except Exception as error:
            if self._is_protocol_error_exception(error):
                raise self._build_protocol_error_from_exception(error) from error
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {error}",
            ) from error
        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_response_stream(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send streaming Responses API request with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            stream_request = dict(request)
            stream_request["stream"] = True

            response_stream = await self._create_with_metadata_fallback(stream_request)
            async for event in response_stream:
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(
                            status_code=499,
                            detail="Request cancelled by client",
                        )

                yield dict(event.model_dump())

        except HTTPException:
            raise
        except AuthenticationError as error:
            raise HTTPException(
                status_code=401,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except RateLimitError as error:
            raise HTTPException(
                status_code=429,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except BadRequestError as error:
            raise HTTPException(
                status_code=400,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except APIError as error:
            if self._is_protocol_error_api_error(error):
                raise self._build_upstream_protocol_error(error) from error

            raw_status_code = getattr(error, "status_code", None)
            raise HTTPException(
                status_code=self._normalize_error_status_code(raw_status_code),
                detail=self.classify_openai_error(str(error)),
            ) from error
        except Exception as error:
            if self._is_protocol_error_exception(error):
                raise self._build_protocol_error_from_exception(error) from error
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {error}",
            ) from error
        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()

        if (
            "unsupported_country_region_territory" in error_str
            or "country, region, or territory not supported" in error_str
        ):
            return (
                "OpenAI API is not available in your region. "
                "Consider using a VPN or Azure OpenAI service."
            )

        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."

        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."

        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."

        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        return str(error_detail)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
