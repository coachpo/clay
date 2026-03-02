from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, Tuple

from fastapi import HTTPException, Request

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequestModel


class LoggerLike(Protocol):
    def info(self, msg: str, *args: Any) -> None: ...

    def warning(self, msg: str, *args: Any) -> None: ...

    def error(self, msg: str, *args: Any) -> None: ...


class OpenAIClientLike(Protocol):
    def cancel_request(self, request_id: str) -> bool: ...


def convert_openai_to_claude_response(
    openai_response: Dict[str, Any],
    original_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
    """Convert OpenAI chat completion response to Anthropic messages response."""
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    content_blocks: List[Dict[str, Any]] = []
    text_content = _extract_text_content(message.get("content"))
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    for tool_call in message.get("tool_calls", []) or []:
        if tool_call.get("type") != Constants.TOOL_FUNCTION:
            continue

        function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
        raw_arguments = function_data.get("arguments", "{}")
        try:
            parsed_arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            parsed_arguments = {"raw_arguments": raw_arguments}

        content_blocks.append(
            {
                "type": Constants.CONTENT_TOOL_USE,
                "id": tool_call.get("id", f"tool_{uuid.uuid4().hex[:24]}"),
                "name": function_data.get("name", ""),
                "input": parsed_arguments,
            }
        )

    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    stop_reason, stop_sequence = _map_finish_reason(
        finish_reason=choice.get("finish_reason") or choice.get("stop_reason"),
        stop_sequence=choice.get("stop_sequence"),
    )

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": _extract_usage(openai_response.get("usage")),
    }


async def convert_openai_streaming_to_claude(
    openai_stream: AsyncGenerator[str, None],
    original_request: ClaudeMessagesRequestModel,
    logger: LoggerLike,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Anthropic streaming format."""
    async for event in _convert_openai_streaming_core(
        openai_stream=openai_stream,
        original_request=original_request,
        logger=logger,
        http_request=None,
        openai_client=None,
        request_id=None,
    ):
        yield event


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream: AsyncGenerator[str, None],
    original_request: ClaudeMessagesRequestModel,
    logger: LoggerLike,
    http_request: Request,
    openai_client: OpenAIClientLike,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Anthropic SSE with disconnect cancellation."""
    async for event in _convert_openai_streaming_core(
        openai_stream=openai_stream,
        original_request=original_request,
        logger=logger,
        http_request=http_request,
        openai_client=openai_client,
        request_id=request_id,
    ):
        yield event


async def _convert_openai_streaming_core(
    openai_stream: AsyncGenerator[str, None],
    original_request: ClaudeMessagesRequestModel,
    logger: LoggerLike,
    http_request: Optional[Request],
    openai_client: Optional[OpenAIClientLike],
    request_id: Optional[str],
) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage_data: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    final_stop_reason = Constants.STOP_END_TURN
    final_stop_sequence: Optional[str] = None

    next_content_index = 0
    text_block_index: Optional[int] = None
    thinking_block_index: Optional[int] = None
    started_block_indexes: List[int] = []
    current_tool_calls: Dict[int, Dict[str, Any]] = {}
    primary_choice_index: Optional[int] = None
    disconnected = False
    yield _sse(
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    try:
        async for line in openai_stream:
            if http_request is not None and request_id is not None:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, cancelling request %s", request_id)
                    if openai_client is not None:
                        openai_client.cancel_request(request_id)
                    disconnected = True
                    break

            if not line or not line.strip() or not line.startswith("data: "):
                continue

            chunk_data = line[6:].strip()
            if chunk_data == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as error:
                logger.warning("Failed to parse chunk '%s': %s", chunk_data, error)
                continue

            chunk_usage = chunk.get("usage")
            if chunk_usage:
                usage_data = _extract_usage(chunk_usage)

            choices = chunk.get("choices", [])
            if not choices:
                continue

            selected_choice = _select_primary_choice(choices, primary_choice_index)
            if selected_choice is None:
                continue
            if primary_choice_index is None:
                selected_index = selected_choice.get("index")
                if isinstance(selected_index, int):
                    primary_choice_index = selected_index
                else:
                    primary_choice_index = 0

            choice = selected_choice
            delta = choice.get("delta") or {}

            content_fragment = delta.get("content")
            if content_fragment is not None:
                text_piece = _extract_text_content(content_fragment)
                if text_piece is None:
                    text_piece = str(content_fragment)

                if text_block_index is None:
                    text_block_index = next_content_index
                    next_content_index += 1
                    started_block_indexes.append(text_block_index)
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": text_block_index,
                            "content_block": {
                                "type": Constants.CONTENT_TEXT,
                                "text": "",
                            },
                        },
                    )

                if text_piece:
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_DELTA,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": text_block_index,
                            "delta": {
                                "type": Constants.DELTA_TEXT,
                                "text": text_piece,
                            },
                        },
                    )

            for tool_call_delta in delta.get("tool_calls") or []:
                call_index = _int_value(tool_call_delta.get("index"), default=0)
                tool_state = current_tool_calls.setdefault(
                    call_index,
                    {
                        "id": None,
                        "name": None,
                        "claude_index": None,
                        "pending_arguments": [],
                    },
                )

                tool_call_id = tool_call_delta.get("id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_state["id"] = tool_call_id

                function_data_raw = tool_call_delta.get(Constants.TOOL_FUNCTION, {})
                function_data = function_data_raw if isinstance(function_data_raw, dict) else {}

                function_name = function_data.get("name")
                if isinstance(function_name, str) and function_name:
                    tool_state["name"] = function_name

                argument_fragment = function_data.get("arguments")
                if argument_fragment is not None:
                    fragment_text = str(argument_fragment)
                    current_index = tool_state.get("claude_index")
                    if current_index is None:
                        tool_state["pending_arguments"].append(fragment_text)
                    elif fragment_text:
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": current_index,
                                "delta": {
                                    "type": Constants.DELTA_INPUT_JSON,
                                    "partial_json": fragment_text,
                                },
                            },
                        )

                if tool_state.get("claude_index") is None and tool_state.get("name"):
                    tool_id = tool_state.get("id") or f"tool_{uuid.uuid4().hex[:24]}"
                    tool_state["id"] = tool_id
                    claude_index = next_content_index
                    tool_state["claude_index"] = claude_index
                    next_content_index += 1
                    started_block_indexes.append(claude_index)
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": claude_index,
                            "content_block": {
                                "type": Constants.CONTENT_TOOL_USE,
                                "id": tool_id,
                                "name": tool_state["name"],
                                "input": {},
                            },
                        },
                    )

                    pending_fragments = tool_state.get("pending_arguments", [])
                    for fragment in pending_fragments:
                        if not fragment:
                            continue
                        yield _sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": claude_index,
                                "delta": {
                                    "type": Constants.DELTA_INPUT_JSON,
                                    "partial_json": fragment,
                                },
                            },
                        )
                    tool_state["pending_arguments"] = []

            reasoning_text = _extract_reasoning_text(delta)
            if reasoning_text:
                if thinking_block_index is None:
                    thinking_block_index = next_content_index
                    next_content_index += 1
                    started_block_indexes.append(thinking_block_index)
                    yield _sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": thinking_block_index,
                            "content_block": {
                                "type": Constants.CONTENT_THINKING,
                                "thinking": "",
                                "signature": "",
                            },
                        },
                    )

                yield _sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": thinking_block_index,
                        "delta": {
                            "type": Constants.DELTA_THINKING,
                            "thinking": reasoning_text,
                        },
                    },
                )

            reasoning_signature = _extract_reasoning_signature(delta)
            if reasoning_signature and thinking_block_index is not None:
                yield _sse(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": thinking_block_index,
                        "delta": {
                            "type": Constants.DELTA_SIGNATURE,
                            "signature": reasoning_signature,
                        },
                    },
                )

            finish_reason = choice.get("finish_reason") or choice.get("stop_reason")
            if finish_reason:
                final_stop_reason, final_stop_sequence = _map_finish_reason(
                    finish_reason=str(finish_reason),
                    stop_sequence=choice.get("stop_sequence"),
                )

    except HTTPException as error:
        if error.status_code == 499:
            logger.info("Request %s was cancelled", request_id)
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "cancelled",
                        "message": "Request was cancelled by client",
                    },
                },
            )
            return
        raise
    except Exception as error:
        logger.error("Streaming error: %s", error)
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Streaming error: {error}",
                },
            },
        )
        return

    if disconnected:
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            },
        )
        return
    for index in started_block_indexes:
        yield _sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": index,
            },
        )

    yield _sse(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": final_stop_sequence,
            },
            "usage": usage_data,
        },
    )
    yield _sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})


def _extract_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}

    normalized: Dict[str, Any] = {
        "input_tokens": _int_value(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
        "output_tokens": _int_value(usage.get("completion_tokens", usage.get("output_tokens", 0))),
    }

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        normalized["cache_read_input_tokens"] = _int_value(prompt_details.get("cached_tokens"))

    if "cache_read_input_tokens" in usage:
        normalized["cache_read_input_tokens"] = _int_value(usage.get("cache_read_input_tokens"))
    if "cache_creation_input_tokens" in usage:
        normalized["cache_creation_input_tokens"] = _int_value(
            usage.get("cache_creation_input_tokens")
        )

    return normalized


def _map_finish_reason(
    finish_reason: Optional[str],
    stop_sequence: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    if finish_reason in {"tool_calls", "function_call"}:
        return Constants.STOP_TOOL_USE, None
    if finish_reason in {"length", "max_tokens"}:
        return Constants.STOP_MAX_TOKENS, None
    if finish_reason == "stop_sequence":
        return Constants.STOP_STOP_SEQUENCE, stop_sequence
    if finish_reason == "pause_turn":
        return Constants.STOP_PAUSE_TURN, None
    if finish_reason == "refusal":
        return Constants.STOP_REFUSAL, None
    if finish_reason == "model_context_window_exceeded":
        return Constants.STOP_MODEL_CONTEXT_WINDOW_EXCEEDED, None
    if finish_reason == "content_filter":
        return Constants.STOP_REFUSAL, None
    if finish_reason == "stop" and stop_sequence:
        return Constants.STOP_STOP_SEQUENCE, stop_sequence
    return Constants.STOP_END_TURN, None


def _extract_text_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text", Constants.CONTENT_TEXT}:
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return str(content)


def _int_value(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _select_primary_choice(
    choices: Any,
    primary_choice_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not isinstance(choices, list):
        return None
    normalized = [choice for choice in choices if isinstance(choice, dict)]
    if not normalized:
        return None

    if primary_choice_index is not None:
        for choice in normalized:
            if choice.get("index") == primary_choice_index:
                return choice
        return None

    for choice in normalized:
        if choice.get("index") == 0:
            return choice

    return None


def _extract_reasoning_text(delta: Dict[str, Any]) -> str:
    candidates = [
        delta.get("reasoning"),
        delta.get("reasoning_content"),
        delta.get("thinking"),
    ]

    for candidate in candidates:
        extracted = _extract_text_content(candidate)
        if extracted:
            return extracted

    return ""


def _extract_reasoning_signature(delta: Dict[str, Any]) -> Optional[str]:
    signature = delta.get("signature")
    if isinstance(signature, str) and signature:
        return signature

    reasoning = delta.get("reasoning")
    if isinstance(reasoning, dict):
        nested = reasoning.get("signature")
        if isinstance(nested, str) and nested:
            return nested

    return None
