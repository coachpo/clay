import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import (
    APIError,
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
        self.base_url = base_url

        self.client: Any
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        self.active_requests: Dict[str, asyncio.Event] = {}

    @staticmethod
    def _is_unsupported_parameter_error(error: BadRequestError, parameter: str) -> bool:
        detail = str(error).lower()
        normalized_parameter = parameter.lower()
        if normalized_parameter not in detail:
            return False
        return "unsupported parameter" in detail or "unknown parameter" in detail

    async def _create_with_metadata_fallback(self, request: Dict[str, Any]) -> Any:
        # Some OpenAI-compatible providers reject selected optional request fields.
        # Retry without the rejected parameter when the provider explicitly reports it.
        retry_request = dict(request)
        fallback_fields = ("metadata", "context_management")

        for _ in range(len(fallback_fields) + 1):
            try:
                return await self.client.responses.create(**retry_request)
            except BadRequestError as error:
                removed_field: Optional[str] = None
                for field in fallback_fields:
                    if field not in retry_request:
                        continue
                    if not self._is_unsupported_parameter_error(error, field):
                        continue
                    next_request = dict(retry_request)
                    next_request.pop(field, None)
                    retry_request = next_request
                    removed_field = field
                    logger.warning(
                        "Upstream rejected optional field '%s'; retrying /v1/responses without it.",
                        field,
                    )
                    break
                if removed_field is not None:
                    continue
                raise

        raise RuntimeError("unexpected fallback loop exit")

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
            status_code = getattr(error, "status_code", 500)
            raise HTTPException(
                status_code=status_code,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except Exception as error:
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
            status_code = getattr(error, "status_code", 500)
            raise HTTPException(
                status_code=status_code,
                detail=self.classify_openai_error(str(error)),
            ) from error
        except Exception as error:
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
