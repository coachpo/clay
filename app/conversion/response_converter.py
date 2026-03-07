from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import Request

from app.core.constants import Constants
from app.models.claude import ClaudeMessagesRequestModel


def convert_responses_to_claude_response(
    responses_api_response: Dict[str, Any],
    original_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []
    saw_tool_use = False

    output_items = extract_responses_output_items(responses_api_response)
    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue

        item_type = output_item.get("type")

        if item_type == "message":
            for text_piece in responses_message_text_parts(output_item):
                if text_piece:
                    content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_piece})
            continue

        if item_type == "function_call":
            saw_tool_use = True
            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": extract_tool_call_id(output_item),
                    "name": str(output_item.get("name") or ""),
                    "input": parse_tool_arguments(output_item.get("arguments", "{}")),
                }
            )
            continue

        if item_type == "reasoning":
            reasoning_text = responses_reasoning_text(output_item)
            if reasoning_text:
                content_blocks.append(
                    {
                        "type": Constants.CONTENT_THINKING,
                        "thinking": reasoning_text,
                        "signature": "",
                    }
                )

    if not content_blocks:
        output_text = extract_text_content(responses_api_response.get("output_text"))
        if output_text:
            content_blocks.append({"type": Constants.CONTENT_TEXT, "text": output_text})

    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    stop_reason, stop_sequence = map_responses_stop_reason(
        responses_api_response,
        saw_tool_use=saw_tool_use,
    )

    return {
        "id": extract_response_message_id(responses_api_response),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": extract_usage(response_usage_payload(responses_api_response)),
    }


async def convert_responses_stream_to_claude(
    responses_stream: AsyncIterator[Any],
    original_request: ClaudeMessagesRequestModel,
    *,
    request: Optional[Request] = None,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage_data: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    final_stop_reason = Constants.STOP_END_TURN
    final_stop_sequence: Optional[str] = None

    next_content_index = 0
    text_block_index: Optional[int] = None
    thinking_block_index: Optional[int] = None
    started_block_indexes: List[int] = []
    saw_tool_use = False

    responses_text_keys_with_deltas: set[str] = set()
    tool_states: Dict[str, Dict[str, Any]] = {}

    def mark_started(index: int) -> None:
        if index not in started_block_indexes:
            started_block_indexes.append(index)

    def emit_text_delta(text: str) -> List[str]:
        nonlocal next_content_index, text_block_index
        if not text:
            return []

        events: List[str] = []
        if text_block_index is None:
            text_block_index = next_content_index
            next_content_index += 1
            mark_started(text_block_index)
            events.append(
                sse(
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
            sse(
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

    def emit_thinking_delta(thinking: str) -> List[str]:
        nonlocal next_content_index, thinking_block_index
        if not thinking:
            return []

        events: List[str] = []
        if thinking_block_index is None:
            thinking_block_index = next_content_index
            next_content_index += 1
            mark_started(thinking_block_index)
            events.append(
                sse(
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
            sse(
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

    def emit_signature_delta(signature: str) -> List[str]:
        if not signature or thinking_block_index is None:
            return []

        return [
            sse(
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

    def tool_state(tool_key: str) -> Dict[str, Any]:
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

    def emit_tool_argument_fragment(tool_state_value: Dict[str, Any], fragment: str) -> List[str]:
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
            sse(
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

    def emit_missing_tool_arguments(
        tool_state_value: Dict[str, Any],
        full_arguments: str,
    ) -> List[str]:
        if not full_arguments:
            return []

        emitted_arguments = str(tool_state_value.get("emitted_arguments", ""))
        if full_arguments == emitted_arguments:
            return []

        if emitted_arguments and full_arguments.startswith(emitted_arguments):
            return emit_tool_argument_fragment(
                tool_state_value,
                full_arguments[len(emitted_arguments) :],
            )

        return emit_tool_argument_fragment(tool_state_value, full_arguments)

    def ensure_tool_block_started(tool_state_value: Dict[str, Any]) -> List[str]:
        nonlocal next_content_index, saw_tool_use

        if tool_state_value.get("claude_index") is not None:
            return []
        if not tool_state_value.get("name"):
            return []

        tool_state_value["id"] = tool_state_value.get("id") or f"tool_{uuid.uuid4().hex[:24]}"
        claude_index = next_content_index
        tool_state_value["claude_index"] = claude_index
        next_content_index += 1
        mark_started(claude_index)
        saw_tool_use = True

        events = [
            sse(
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
            events.extend(emit_tool_argument_fragment(tool_state_value, str(pending_fragment)))
        tool_state_value["pending_arguments"] = []
        return events

    yield sse(
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
    yield sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    async for raw_event in responses_stream:
        if request is not None and await request.is_disconnected():
            return

        for stream_event in normalize_stream_payload(raw_event):
            event_type = stream_event.get("type")

            if event_type == "error":
                yield sse(Constants.EVENT_ERROR, normalize_responses_error_payload(stream_event))
                return

            if not isinstance(event_type, str) or not event_type.startswith("response."):
                continue

            if event_type in {"response.created", "response.in_progress"}:
                response_payload = stream_event.get("response")
                if isinstance(response_payload, dict):
                    usage_data = extract_usage(response_usage_payload(response_payload))
                continue

            if event_type == "response.output_text.delta":
                text_key = responses_text_key(stream_event)
                text_piece = str(stream_event.get("delta") or "")
                if text_piece:
                    responses_text_keys_with_deltas.add(text_key)
                    for event_line in emit_text_delta(text_piece):
                        yield event_line
                continue

            if event_type == "response.output_text.done":
                text_key = responses_text_key(stream_event)
                if text_key in responses_text_keys_with_deltas:
                    continue
                done_text = extract_text_content(stream_event.get("text"))
                if done_text:
                    for event_line in emit_text_delta(done_text):
                        yield event_line
                continue

            if event_type in {"response.output_item.added", "response.output_item.done"}:
                item = stream_event.get("item")
                if not isinstance(item, dict):
                    continue

                output_index = responses_output_index(stream_event)
                is_done = event_type.endswith(".done")
                item_type = item.get("type")

                if item_type == "function_call":
                    tool_state_value = tool_state(
                        responses_tool_key(
                            output_index=output_index,
                            item_id=item.get("id"),
                            call_id=item.get("call_id"),
                        )
                    )
                    tool_state_value["id"] = extract_tool_call_id(item)
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        tool_state_value["name"] = name

                    for event_line in ensure_tool_block_started(tool_state_value):
                        yield event_line

                    arguments = item.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        if is_done:
                            for event_line in emit_missing_tool_arguments(
                                tool_state_value,
                                arguments,
                            ):
                                yield event_line
                        else:
                            for event_line in emit_tool_argument_fragment(
                                tool_state_value,
                                arguments,
                            ):
                                yield event_line
                    continue

                if item_type == "reasoning":
                    for event_line in emit_thinking_delta(responses_reasoning_text(item)):
                        yield event_line
                    continue

                if item_type == "message" and is_done and text_block_index is None:
                    for text_piece in responses_message_text_parts(item):
                        for event_line in emit_text_delta(text_piece):
                            yield event_line
                    continue

                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                output_index = responses_output_index(stream_event)
                item_id = stream_event.get("item_id")
                call_id = stream_event.get("call_id")
                tool_state_value = tool_state(
                    responses_tool_key(
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
                    for event_line in emit_tool_argument_fragment(tool_state_value, delta_fragment):
                        yield event_line
                else:
                    full_arguments = str(stream_event.get("arguments") or "")
                    for event_line in emit_missing_tool_arguments(tool_state_value, full_arguments):
                        yield event_line

                for event_line in ensure_tool_block_started(tool_state_value):
                    yield event_line

                continue

            if event_type == "response.reasoning.delta":
                reasoning_delta = str(stream_event.get("delta") or "")
                for event_line in emit_thinking_delta(reasoning_delta):
                    yield event_line
                for event_line in emit_signature_delta(str(stream_event.get("signature") or "")):
                    yield event_line
                continue

            if event_type in {"response.completed", "response.incomplete"}:
                response_payload = stream_event.get("response")
                if isinstance(response_payload, dict):
                    usage_data = extract_usage(response_usage_payload(response_payload))
                    final_stop_reason, final_stop_sequence = map_responses_stop_reason(
                        response_payload,
                        saw_tool_use=saw_tool_use,
                    )
                break

            if event_type == "response.failed":
                yield sse(Constants.EVENT_ERROR, normalize_responses_error_payload(stream_event))
                return

    for index in started_block_indexes:
        yield sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": index,
            },
        )

    yield sse(
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
    yield sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})


def normalize_stream_payload(raw_event: Any) -> List[Dict[str, Any]]:
    if hasattr(raw_event, "model_dump"):
        raw_event = raw_event.model_dump(mode="json", warnings=False)

    if isinstance(raw_event, dict):
        if isinstance(raw_event.get("type"), str):
            return [raw_event]
        return []

    if not isinstance(raw_event, str):
        return []

    line = raw_event.strip()
    if not line or not line.startswith("data: "):
        return []

    chunk_data = line[6:].strip()
    if chunk_data == "[DONE]":
        return []

    try:
        payload = json.loads(chunk_data)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        return [payload]
    return []


def responses_output_index(event: Dict[str, Any]) -> Optional[int]:
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


def responses_text_key(event: Dict[str, Any]) -> str:
    return f"{event.get('output_index')}:{event.get('content_index')}:{event.get('item_id')}"


def responses_tool_key(
    *,
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


def normalize_responses_error_payload(event: Dict[str, Any]) -> Dict[str, Any]:
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


def extract_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}

    normalized: Dict[str, Any] = {
        "input_tokens": int_value(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
        "output_tokens": int_value(usage.get("completion_tokens", usage.get("output_tokens", 0))),
    }

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        normalized["total_tokens"] = int_value(total_tokens)

    return normalized


def map_responses_stop_reason(
    response_payload: Dict[str, Any],
    *,
    saw_tool_use: bool,
) -> Tuple[str, Optional[str]]:
    if saw_tool_use:
        return Constants.STOP_TOOL_USE, None

    output_items = extract_responses_output_items(response_payload)
    if any(isinstance(item, dict) and item.get("type") == "function_call" for item in output_items):
        return Constants.STOP_TOOL_USE, None

    finish_reason = response_payload.get("stop_reason") or response_payload.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return map_finish_reason(finish_reason=finish_reason, stop_sequence=None)

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


def map_finish_reason(
    *,
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


def extract_responses_output_items(responses_api_response: Dict[str, Any]) -> List[Any]:
    output_items = responses_api_response.get("output")
    if isinstance(output_items, list):
        return output_items

    nested_response = responses_api_response.get("response")
    if isinstance(nested_response, dict):
        nested_output = nested_response.get("output")
        if isinstance(nested_output, list):
            return nested_output

    return []


def extract_response_message_id(responses_api_response: Dict[str, Any]) -> str:
    for output_item in extract_responses_output_items(responses_api_response):
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") == "message":
            item_id = output_item.get("id")
            if isinstance(item_id, str) and item_id:
                return item_id

    response_id = responses_api_response.get("id")
    if isinstance(response_id, str) and response_id:
        return response_id

    nested_response = responses_api_response.get("response")
    if isinstance(nested_response, dict):
        nested_id = nested_response.get("id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id

    return f"msg_{uuid.uuid4().hex[:24]}"


def response_usage_payload(responses_api_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = responses_api_response.get("usage")
    if isinstance(usage, dict):
        return usage

    nested_response = responses_api_response.get("response")
    if isinstance(nested_response, dict):
        nested_usage = nested_response.get("usage")
        if isinstance(nested_usage, dict):
            return nested_usage

    return None


def responses_message_text_parts(message_item: Dict[str, Any]) -> List[str]:
    content = message_item.get("content")
    if not isinstance(content, list):
        return []

    text_parts: List[str] = []
    for content_part in content:
        if not isinstance(content_part, dict):
            extracted = extract_text_content(content_part)
            if extracted:
                text_parts.append(extracted)
            continue

        part_type = content_part.get("type")
        if part_type in {"output_text", Constants.CONTENT_TEXT}:
            text_value = extract_text_content(content_part.get("text"))
            if text_value:
                text_parts.append(text_value)
            continue

        if part_type in {"refusal", "output_refusal"}:
            refusal_text = extract_text_content(content_part.get("refusal"))
            if refusal_text:
                text_parts.append(refusal_text)
            continue

        extracted = extract_text_content(content_part)
        if extracted:
            text_parts.append(extracted)

    return text_parts


def responses_reasoning_text(reasoning_item: Dict[str, Any]) -> str:
    summary_text = extract_text_content(reasoning_item.get("summary"))
    if summary_text:
        return summary_text

    content_text = extract_text_content(reasoning_item.get("content"))
    if content_text:
        return content_text

    return ""


def extract_tool_call_id(tool_item: Dict[str, Any]) -> str:
    call_id = tool_item.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id

    item_id = tool_item.get("id")
    if isinstance(item_id, str) and item_id:
        return item_id

    return f"tool_{uuid.uuid4().hex[:24]}"


def parse_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
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


def extract_text_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            extracted = extract_text_content(item)
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


def int_value(value: Any, default: int = 0) -> int:
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


def sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
