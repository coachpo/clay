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
    """Convert OpenAI response payloads into Anthropic messages format."""
    if _looks_like_chat_completion_response(openai_response):
        return _convert_chat_completion_to_claude_response(openai_response, original_request)
    return _convert_responses_api_to_claude_response(openai_response, original_request)


def _looks_like_chat_completion_response(openai_response: Dict[str, Any]) -> bool:
    return isinstance(openai_response.get("choices"), list)


def _convert_chat_completion_to_claude_response(
    openai_response: Dict[str, Any],
    original_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
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

        function_data_raw = tool_call.get(Constants.TOOL_FUNCTION, {})
        function_data = function_data_raw if isinstance(function_data_raw, dict) else {}
        content_blocks.append(
            {
                "type": Constants.CONTENT_TOOL_USE,
                "id": tool_call.get("id", f"tool_{uuid.uuid4().hex[:24]}"),
                "name": str(function_data.get("name") or ""),
                "input": _parse_tool_arguments(function_data.get("arguments", "{}")),
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
        "usage": _extract_usage(_response_usage_payload(openai_response)),
    }


def _convert_responses_api_to_claude_response(
    openai_response: Dict[str, Any],
    original_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []
    saw_tool_use = False

    output_items = _extract_responses_output_items(openai_response)
    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue

        item_type = output_item.get("type")

        if item_type == "message":
            for text_piece in _responses_message_text_parts(output_item):
                if text_piece:
                    content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_piece})
            continue

        if item_type == "function_call":
            saw_tool_use = True
            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": _extract_tool_call_id(output_item),
                    "name": str(output_item.get("name") or ""),
                    "input": _parse_tool_arguments(output_item.get("arguments", "{}")),
                }
            )
            continue

        if item_type == "reasoning":
            reasoning_text = _responses_reasoning_text(output_item)
            if reasoning_text:
                content_blocks.append(
                    {
                        "type": Constants.CONTENT_THINKING,
                        "thinking": reasoning_text,
                        "signature": "",
                    }
                )

    if not content_blocks:
        output_text = _extract_text_content(openai_response.get("output_text"))
        if output_text:
            content_blocks.append({"type": Constants.CONTENT_TEXT, "text": output_text})

    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    stop_reason, stop_sequence = _map_responses_stop_reason(openai_response, saw_tool_use)

    return {
        "id": _extract_response_message_id(openai_response),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": _extract_usage(_response_usage_payload(openai_response)),
    }


def _extract_responses_output_items(openai_response: Dict[str, Any]) -> List[Any]:
    output_items = openai_response.get("output")
    if isinstance(output_items, list):
        return output_items

    nested_response = openai_response.get("response")
    if isinstance(nested_response, dict):
        nested_output = nested_response.get("output")
        if isinstance(nested_output, list):
            return nested_output

    return []


def _extract_response_message_id(openai_response: Dict[str, Any]) -> str:
    for output_item in _extract_responses_output_items(openai_response):
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") == "message":
            item_id = output_item.get("id")
            if isinstance(item_id, str) and item_id:
                return item_id

    response_id = openai_response.get("id")
    if isinstance(response_id, str) and response_id:
        return response_id

    nested_response = openai_response.get("response")
    if isinstance(nested_response, dict):
        nested_id = nested_response.get("id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id

    return f"msg_{uuid.uuid4().hex[:24]}"


def _response_usage_payload(openai_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = openai_response.get("usage")
    if isinstance(usage, dict):
        return usage

    nested_response = openai_response.get("response")
    if isinstance(nested_response, dict):
        nested_usage = nested_response.get("usage")
        if isinstance(nested_usage, dict):
            return nested_usage

    return None


def _responses_message_text_parts(message_item: Dict[str, Any]) -> List[str]:
    content = message_item.get("content")
    if not isinstance(content, list):
        return []

    text_parts: List[str] = []
    for content_part in content:
        if not isinstance(content_part, dict):
            extracted = _extract_text_content(content_part)
            if extracted:
                text_parts.append(extracted)
            continue

        part_type = content_part.get("type")
        if part_type in {"output_text", Constants.CONTENT_TEXT}:
            text_value = _extract_text_content(content_part.get("text"))
            if text_value:
                text_parts.append(text_value)
            continue

        if part_type in {"refusal", "output_refusal"}:
            refusal_text = _extract_text_content(content_part.get("refusal"))
            if refusal_text:
                text_parts.append(refusal_text)
            continue

        extracted = _extract_text_content(content_part)
        if extracted:
            text_parts.append(extracted)

    return text_parts


def _responses_reasoning_text(reasoning_item: Dict[str, Any]) -> str:
    summary_text = _extract_text_content(reasoning_item.get("summary"))
    if summary_text:
        return summary_text

    content_text = _extract_text_content(reasoning_item.get("content"))
    if content_text:
        return content_text

    return ""


def _extract_tool_call_id(tool_item: Dict[str, Any]) -> str:
    call_id = tool_item.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id

    item_id = tool_item.get("id")
    if isinstance(item_id, str) and item_id:
        return item_id

    return f"tool_{uuid.uuid4().hex[:24]}"


def _parse_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw_arguments": raw_arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    if raw_arguments is None:
        return {}

    return {"value": raw_arguments}


async def convert_openai_streaming_to_claude(
    openai_stream: AsyncGenerator[Any, None],
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
    openai_stream: AsyncGenerator[Any, None],
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
    openai_stream: AsyncGenerator[Any, None],
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
    primary_choice_index: Optional[int] = None
    saw_tool_use = False
    disconnected = False

    responses_text_keys_with_deltas: set[str] = set()
    tool_states: Dict[str, Dict[str, Any]] = {}

    def _mark_started(index: int) -> None:
        if index not in started_block_indexes:
            started_block_indexes.append(index)

    def _emit_text_delta(text: str) -> List[str]:
        nonlocal next_content_index, text_block_index
        if not text:
            return []

        events: List[str] = []
        if text_block_index is None:
            text_block_index = next_content_index
            next_content_index += 1
            _mark_started(text_block_index)
            events.append(
                _sse(
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
            )

        events.append(
            _sse(
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": text_block_index,
                    "delta": {
                        "type": Constants.DELTA_TEXT,
                        "text": text,
                    },
                },
            )
        )
        return events

    def _emit_thinking_delta(thinking: str) -> List[str]:
        nonlocal next_content_index, thinking_block_index
        if not thinking:
            return []

        events: List[str] = []
        if thinking_block_index is None:
            thinking_block_index = next_content_index
            next_content_index += 1
            _mark_started(thinking_block_index)
            events.append(
                _sse(
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
            )

        events.append(
            _sse(
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": thinking_block_index,
                    "delta": {
                        "type": Constants.DELTA_THINKING,
                        "thinking": thinking,
                    },
                },
            )
        )
        return events

    def _emit_signature_delta(signature: str) -> List[str]:
        if not signature or thinking_block_index is None:
            return []

        return [
            _sse(
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": thinking_block_index,
                    "delta": {
                        "type": Constants.DELTA_SIGNATURE,
                        "signature": signature,
                    },
                },
            )
        ]

    def _tool_state(tool_key: str) -> Dict[str, Any]:
        return tool_states.setdefault(
            tool_key,
            {
                "id": None,
                "name": None,
                "claude_index": None,
                "pending_arguments": [],
                "emitted_arguments": "",
            },
        )

    def _emit_tool_argument_fragment(tool_state_value: Dict[str, Any], fragment: str) -> List[str]:
        if not fragment:
            return []

        claude_index = tool_state_value.get("claude_index")
        if claude_index is None:
            tool_state_value.setdefault("pending_arguments", []).append(fragment)
            return []

        tool_state_value["emitted_arguments"] = (
            str(tool_state_value.get("emitted_arguments", "")) + fragment
        )

        return [
            _sse(
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
        ]

    def _emit_missing_tool_arguments(
        tool_state_value: Dict[str, Any], full_arguments: str
    ) -> List[str]:
        if not full_arguments:
            return []

        emitted_arguments = str(tool_state_value.get("emitted_arguments", ""))
        if full_arguments == emitted_arguments:
            return []

        if emitted_arguments and full_arguments.startswith(emitted_arguments):
            return _emit_tool_argument_fragment(
                tool_state_value,
                full_arguments[len(emitted_arguments) :],
            )

        return _emit_tool_argument_fragment(tool_state_value, full_arguments)

    def _ensure_tool_block_started(tool_state_value: Dict[str, Any]) -> List[str]:
        nonlocal next_content_index, saw_tool_use

        if tool_state_value.get("claude_index") is not None:
            return []
        if not tool_state_value.get("name"):
            return []

        tool_state_value["id"] = tool_state_value.get("id") or f"tool_{uuid.uuid4().hex[:24]}"
        claude_index = next_content_index
        tool_state_value["claude_index"] = claude_index
        next_content_index += 1
        _mark_started(claude_index)
        saw_tool_use = True

        events = [
            _sse(
                Constants.EVENT_CONTENT_BLOCK_START,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_START,
                    "index": claude_index,
                    "content_block": {
                        "type": Constants.CONTENT_TOOL_USE,
                        "id": tool_state_value["id"],
                        "name": tool_state_value["name"],
                        "input": {},
                    },
                },
            )
        ]

        pending_arguments = tool_state_value.get("pending_arguments", [])
        for pending_fragment in pending_arguments:
            events.extend(_emit_tool_argument_fragment(tool_state_value, str(pending_fragment)))
        tool_state_value["pending_arguments"] = []
        return events

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
        stream_finished = False
        async for raw_event in openai_stream:
            if (
                http_request is not None
                and request_id is not None
                and await http_request.is_disconnected()
            ):
                logger.info("Client disconnected, cancelling request %s", request_id)
                if openai_client is not None:
                    openai_client.cancel_request(request_id)
                disconnected = True
                break

            normalized_events = _normalize_stream_payload(raw_event, logger)
            if not normalized_events:
                continue

            for stream_event in normalized_events:
                event_type = stream_event.get("type")

                if event_type == "_done":
                    stream_finished = True
                    break

                if event_type == "_legacy_chunk":
                    chunk = stream_event.get("chunk")
                    if not isinstance(chunk, dict):
                        continue

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage_data = _extract_usage(chunk_usage)

                    choices = chunk.get("choices", [])
                    selected_choice = _select_primary_choice(choices, primary_choice_index)
                    if selected_choice is None:
                        continue

                    if primary_choice_index is None:
                        selected_index = selected_choice.get("index")
                        primary_choice_index = (
                            selected_index if isinstance(selected_index, int) else 0
                        )

                    delta = selected_choice.get("delta") or {}

                    content_fragment = delta.get("content")
                    if content_fragment is not None:
                        text_piece = _extract_text_content(content_fragment)
                        if text_piece is None:
                            text_piece = str(content_fragment)
                        for event_line in _emit_text_delta(text_piece):
                            yield event_line

                    for tool_call_delta in delta.get("tool_calls") or []:
                        call_index = _int_value(tool_call_delta.get("index"), default=0)
                        tool_state_value = _tool_state(f"legacy:{call_index}")

                        tool_call_id = tool_call_delta.get("id")
                        if isinstance(tool_call_id, str) and tool_call_id:
                            tool_state_value["id"] = tool_call_id

                        function_raw = tool_call_delta.get(Constants.TOOL_FUNCTION, {})
                        function_data = function_raw if isinstance(function_raw, dict) else {}

                        function_name = function_data.get("name")
                        if isinstance(function_name, str) and function_name:
                            tool_state_value["name"] = function_name

                        arguments_fragment = function_data.get("arguments")
                        if arguments_fragment is not None:
                            for event_line in _emit_tool_argument_fragment(
                                tool_state_value, str(arguments_fragment)
                            ):
                                yield event_line

                        for event_line in _ensure_tool_block_started(tool_state_value):
                            yield event_line

                    for event_line in _emit_thinking_delta(_extract_reasoning_text(delta)):
                        yield event_line
                    for event_line in _emit_signature_delta(
                        _extract_reasoning_signature(delta) or ""
                    ):
                        yield event_line

                    finish_reason = selected_choice.get("finish_reason") or selected_choice.get(
                        "stop_reason"
                    )
                    if finish_reason:
                        final_stop_reason, final_stop_sequence = _map_finish_reason(
                            finish_reason=str(finish_reason),
                            stop_sequence=selected_choice.get("stop_sequence"),
                        )

                    continue

                if event_type == "error":
                    yield _sse("error", _normalize_responses_error_payload(stream_event))
                    return

                if not isinstance(event_type, str) or not event_type.startswith("response."):
                    continue

                if event_type in {"response.created", "response.in_progress"}:
                    response_payload = stream_event.get("response")
                    if isinstance(response_payload, dict):
                        usage_data = _extract_usage(_response_usage_payload(response_payload))
                    continue

                if event_type == "response.output_text.delta":
                    text_key = _responses_text_key(stream_event)
                    text_piece = str(stream_event.get("delta") or "")
                    if text_piece:
                        responses_text_keys_with_deltas.add(text_key)
                        for event_line in _emit_text_delta(text_piece):
                            yield event_line
                    continue

                if event_type == "response.output_text.done":
                    text_key = _responses_text_key(stream_event)
                    if text_key in responses_text_keys_with_deltas:
                        continue
                    text_piece = _extract_text_content(stream_event.get("text"))
                    if text_piece:
                        for event_line in _emit_text_delta(text_piece):
                            yield event_line
                    continue

                if event_type in {"response.output_item.added", "response.output_item.done"}:
                    item = stream_event.get("item")
                    if not isinstance(item, dict):
                        continue

                    output_index = _responses_output_index(stream_event)
                    is_done = event_type.endswith(".done")
                    item_type = item.get("type")

                    if item_type == "function_call":
                        tool_state_value = _tool_state(
                            _responses_tool_key(
                                output_index=output_index,
                                item_id=item.get("id"),
                                call_id=item.get("call_id"),
                            )
                        )
                        tool_state_value["id"] = _extract_tool_call_id(item)
                        name = item.get("name")
                        if isinstance(name, str) and name:
                            tool_state_value["name"] = name

                        for event_line in _ensure_tool_block_started(tool_state_value):
                            yield event_line

                        arguments = item.get("arguments")
                        if isinstance(arguments, str) and arguments:
                            if is_done:
                                for event_line in _emit_missing_tool_arguments(
                                    tool_state_value, arguments
                                ):
                                    yield event_line
                            else:
                                for event_line in _emit_tool_argument_fragment(
                                    tool_state_value, arguments
                                ):
                                    yield event_line
                        continue

                    if item_type == "reasoning":
                        for event_line in _emit_thinking_delta(_responses_reasoning_text(item)):
                            yield event_line
                        continue

                    if item_type == "message" and is_done and text_block_index is None:
                        for text_piece in _responses_message_text_parts(item):
                            for event_line in _emit_text_delta(text_piece):
                                yield event_line
                        continue

                    continue

                if event_type in {
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                }:
                    output_index = _responses_output_index(stream_event)
                    item_id = stream_event.get("item_id")
                    call_id = stream_event.get("call_id")
                    tool_state_value = _tool_state(
                        _responses_tool_key(
                            output_index=output_index,
                            item_id=item_id,
                            call_id=call_id,
                        )
                    )

                    if isinstance(call_id, str) and call_id:
                        tool_state_value["id"] = call_id
                    if isinstance(item_id, str) and item_id and not tool_state_value.get("id"):
                        tool_state_value["id"] = item_id

                    name = stream_event.get("name")
                    if isinstance(name, str) and name:
                        tool_state_value["name"] = name

                    if event_type.endswith(".delta"):
                        delta_fragment = str(stream_event.get("delta") or "")
                        for event_line in _emit_tool_argument_fragment(
                            tool_state_value, delta_fragment
                        ):
                            yield event_line
                    else:
                        full_arguments = str(stream_event.get("arguments") or "")
                        for event_line in _emit_missing_tool_arguments(
                            tool_state_value, full_arguments
                        ):
                            yield event_line

                    for event_line in _ensure_tool_block_started(tool_state_value):
                        yield event_line

                    continue

                if event_type in {"response.completed", "response.incomplete"}:
                    response_payload = stream_event.get("response")
                    if isinstance(response_payload, dict):
                        usage_data = _extract_usage(_response_usage_payload(response_payload))
                        final_stop_reason, final_stop_sequence = _map_responses_stop_reason(
                            response_payload,
                            saw_tool_use=saw_tool_use,
                        )
                    else:
                        final_stop_reason = (
                            Constants.STOP_TOOL_USE if saw_tool_use else Constants.STOP_END_TURN
                        )
                        final_stop_sequence = None
                    stream_finished = True
                    break

                if event_type == "response.failed":
                    yield _sse("error", _normalize_responses_error_payload(stream_event))
                    return

            if stream_finished:
                break

    except HTTPException as error:
        message = str(error.detail)
        logger.error("Streaming HTTP error: %s", message)
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
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message,
                },
            },
        )
        return
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


def _normalize_stream_payload(raw_event: Any, logger: LoggerLike) -> List[Dict[str, Any]]:
    if isinstance(raw_event, dict):
        if isinstance(raw_event.get("type"), str):
            return [raw_event]
        return [{"type": "_legacy_chunk", "chunk": raw_event}]

    if not isinstance(raw_event, str):
        return []

    line = raw_event.strip()
    if not line or not line.startswith("data: "):
        return []

    chunk_data = line[6:].strip()
    if chunk_data == "[DONE]":
        return [{"type": "_done"}]

    try:
        payload = json.loads(chunk_data)
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse stream payload '%s': %s", chunk_data, error)
        return []

    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("type"), str):
        return [payload]
    return [{"type": "_legacy_chunk", "chunk": payload}]


def _responses_output_index(event: Dict[str, Any]) -> Optional[int]:
    value = event.get("output_index")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _responses_text_key(event: Dict[str, Any]) -> str:
    return f"{event.get('output_index')}:{event.get('content_index')}:{event.get('item_id')}"


def _responses_tool_key(
    output_index: Optional[int],
    item_id: Any,
    call_id: Any,
) -> str:
    if output_index is not None:
        return f"output:{output_index}"
    if isinstance(item_id, str) and item_id:
        return f"item:{item_id}"
    if isinstance(call_id, str) and call_id:
        return f"call:{call_id}"
    return "output:0"


def _normalize_responses_error_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    if event.get("type") == "error":
        error_block = event.get("error")
        if isinstance(error_block, dict):
            return {
                "type": "error",
                "error": {
                    "type": str(error_block.get("type") or "api_error"),
                    "message": str(error_block.get("message") or "Streaming error"),
                },
            }

    response_block = event.get("response")
    if isinstance(response_block, dict):
        error_block = response_block.get("error")
        if isinstance(error_block, dict):
            return {
                "type": "error",
                "error": {
                    "type": str(error_block.get("type") or "api_error"),
                    "message": str(error_block.get("message") or "Streaming error"),
                },
            }

    return {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "Streaming error",
        },
    }


def _extract_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}

    normalized: Dict[str, Any] = {
        "input_tokens": _int_value(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
        "output_tokens": _int_value(usage.get("completion_tokens", usage.get("output_tokens", 0))),
    }

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        normalized["total_tokens"] = _int_value(total_tokens)

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        normalized["cache_read_input_tokens"] = _int_value(prompt_details.get("cached_tokens"))

    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict) and "cached_tokens" in input_details:
        normalized["cache_read_input_tokens"] = _int_value(input_details.get("cached_tokens"))

    if "cache_read_input_tokens" in usage:
        normalized["cache_read_input_tokens"] = _int_value(usage.get("cache_read_input_tokens"))
    if "cache_creation_input_tokens" in usage:
        normalized["cache_creation_input_tokens"] = _int_value(
            usage.get("cache_creation_input_tokens")
        )

    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict) and "reasoning_tokens" in completion_details:
        normalized["reasoning_output_tokens"] = _int_value(
            completion_details.get("reasoning_tokens")
        )

    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict) and "reasoning_tokens" in output_details:
        normalized["reasoning_output_tokens"] = _int_value(output_details.get("reasoning_tokens"))

    return normalized


def _map_responses_stop_reason(
    response_payload: Dict[str, Any],
    saw_tool_use: bool,
) -> Tuple[str, Optional[str]]:
    if saw_tool_use:
        return Constants.STOP_TOOL_USE, None

    output_items = _extract_responses_output_items(response_payload)
    if any(isinstance(item, dict) and item.get("type") == "function_call" for item in output_items):
        return Constants.STOP_TOOL_USE, None

    finish_reason = response_payload.get("stop_reason") or response_payload.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return _map_finish_reason(finish_reason=finish_reason, stop_sequence=None)

    status = response_payload.get("status")
    incomplete_details = response_payload.get("incomplete_details")
    incomplete_reason: Optional[str] = None
    if isinstance(incomplete_details, dict):
        reason_value = incomplete_details.get("reason")
        if isinstance(reason_value, str):
            incomplete_reason = reason_value

    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return Constants.STOP_MAX_TOKENS, None
        if incomplete_reason == "content_filter":
            return Constants.STOP_REFUSAL, None

    if status == "failed" or response_payload.get("error"):
        return Constants.STOP_ERROR, None

    return Constants.STOP_END_TURN, None


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
            extracted = _extract_text_content(item)
            if extracted:
                text_parts.append(extracted)
        return "".join(text_parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        refusal_value = content.get("refusal")
        if isinstance(refusal_value, str):
            return refusal_value
        return None
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
