import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)


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

    async def create_chat_completion(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            completion_task = asyncio.create_task(self.client.chat.completions.create(**request))

            if request_id:
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for pending_task in pending:
                    pending_task.cancel()
                    try:
                        await pending_task
                    except asyncio.CancelledError:
                        pass

                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(
                        status_code=499,
                        detail="Request cancelled by client",
                    )

                completion = await completion_task
            else:
                completion = await completion_task

            return dict(completion.model_dump())

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

    async def create_chat_completion_stream(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            request["stream"] = True
            stream_options = request.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
                request["stream_options"] = stream_options
            stream_options["include_usage"] = True

            streaming_completion = await self.client.chat.completions.create(**request)
            async for chunk in streaming_completion:
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(
                            status_code=499,
                            detail="Request cancelled by client",
                        )

                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            yield "data: [DONE]"

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
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL " "configuration."

        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        return str(error_detail)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
