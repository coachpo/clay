from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Request

from app.core.constants import Constants
from app.models.claude import ClaudeMessagesRequestModel


def convert_openai_to_claude_response(
    openai_response: Dict[str, Any],
    original_request: ClaudeMessagesRequestModel,
) -> Dict[str, Any]:
    choices = openai_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices in LiteLLM response")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message_payload = message if isinstance(message, dict) else {}

    content_blocks: List[Dict[str, Any]] = []
    text_content = extract_text_content(message_payload.get("content"))
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    tool_calls = message_payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or tool_call.get("type") != Constants.TOOL_FUNCTION:
                continue

            function_payload = tool_call.get("function")
            function_data = function_payload if isinstance(function_payload, dict) else {}
            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id") or f"tool_{uuid.uuid4().hex[:24]}",
                    "name": str(function_data.get("name") or ""),
                    "input": parse_tool_arguments(function_data.get("arguments")),
                }
            )

    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    stop_reason, stop_sequence = map_finish_reason(
        finish_reason=choice.get("finish_reason") or choice.get("stop_reason"),
        stop_sequence=choice.get("stop_sequence"),
    )

    return {
        "id": openai_response.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": extract_usage(openai_response.get("usage")),
    }


async def convert_openai_streaming_to_claude(
    openai_stream: AsyncGenerator[Any, None],
    original_request: ClaudeMessagesRequestModel,
    *,
    request: Optional[Request] = None,
) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    next_content_index = 0
    final_stop_reason = Constants.STOP_END_TURN
    final_stop_sequence: Optional[str] = None
    started_indexes: List[int] = []
    text_index: Optional[int] = None
    tool_states: Dict[str, Dict[str, Any]] = {}

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
                "usage": usage,
            },
        },
    )
    yield sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    async for raw_chunk in openai_stream:
        if request is not None and await request.is_disconnected():
            return

        chunk = raw_chunk.model_dump() if hasattr(raw_chunk, "model_dump") else raw_chunk
        if not isinstance(chunk, dict):
            continue

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = extract_usage(chunk_usage)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta")
        delta_payload = delta if isinstance(delta, dict) else {}

        content_fragment = delta_payload.get("content")
        text_piece = extract_text_content(content_fragment)
        if text_piece:
            if text_index is None:
                text_index = next_content_index
                next_content_index += 1
                started_indexes.append(text_index)
                yield sse(
                    Constants.EVENT_CONTENT_BLOCK_START,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_START,
                        "index": text_index,
                        "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
                    },
                )

            yield sse(
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": text_index,
                    "delta": {"type": Constants.DELTA_TEXT, "text": text_piece},
                },
            )

        tool_call_deltas = delta_payload.get("tool_calls")
        if isinstance(tool_call_deltas, list):
            for tool_call_delta in tool_call_deltas:
                if not isinstance(tool_call_delta, dict):
                    continue

                tool_key = str(tool_call_delta.get("index", len(tool_states)))
                tool_state = tool_states.setdefault(
                    tool_key,
                    {
                        "claude_index": None,
                        "id": None,
                        "name": None,
                        "pending_arguments": [],
                    },
                )

                tool_call_id = tool_call_delta.get("id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_state["id"] = tool_call_id

                function_payload = tool_call_delta.get("function")
                function_data = function_payload if isinstance(function_payload, dict) else {}
                function_name = function_data.get("name")
                if isinstance(function_name, str) and function_name:
                    tool_state["name"] = function_name

                arguments = function_data.get("arguments")
                if isinstance(arguments, str) and arguments:
                    tool_state["pending_arguments"].append(arguments)

                if tool_state["claude_index"] is None and tool_state["name"]:
                    tool_index = next_content_index
                    tool_state["claude_index"] = tool_index
                    next_content_index += 1
                    started_indexes.append(tool_index)
                    yield sse(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": tool_index,
                            "content_block": {
                                "type": Constants.CONTENT_TOOL_USE,
                                "id": tool_state["id"] or f"tool_{uuid.uuid4().hex[:24]}",
                                "name": tool_state["name"],
                                "input": {},
                            },
                        },
                    )

                if tool_state["claude_index"] is not None and tool_state["pending_arguments"]:
                    for fragment in tool_state["pending_arguments"]:
                        yield sse(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": tool_state["claude_index"],
                                "delta": {
                                    "type": Constants.DELTA_INPUT_JSON,
                                    "partial_json": fragment,
                                },
                            },
                        )
                    tool_state["pending_arguments"] = []

        finish_reason = choice.get("finish_reason") or choice.get("stop_reason")
        if finish_reason:
            final_stop_reason, final_stop_sequence = map_finish_reason(
                finish_reason=finish_reason,
                stop_sequence=choice.get("stop_sequence"),
            )

    for index in started_indexes:
        yield sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": index},
        )

    yield sse(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": final_stop_sequence,
            },
            "usage": usage,
        },
    )
    yield sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})


def extract_text_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text_value = item.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
        return "".join(text_parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return str(content)


def extract_usage(usage: Any) -> Dict[str, int]:
    payload = usage if isinstance(usage, dict) else {}
    input_tokens = payload.get("prompt_tokens", payload.get("input_tokens", 0))
    output_tokens = payload.get("completion_tokens", payload.get("output_tokens", 0))
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }


def map_finish_reason(
    finish_reason: Any,
    stop_sequence: Any,
) -> Tuple[str, Optional[str]]:
    normalized_reason = str(finish_reason or "").strip().lower()

    if normalized_reason in {"length", "max_tokens"}:
        return Constants.STOP_MAX_TOKENS, None
    if normalized_reason in {"tool_calls", "function_call"}:
        return Constants.STOP_TOOL_USE, None
    if normalized_reason in {"content_filter", "refusal"}:
        return Constants.STOP_REFUSAL, None
    if normalized_reason == "stop" and isinstance(stop_sequence, str) and stop_sequence:
        return Constants.STOP_STOP_SEQUENCE, stop_sequence
    if normalized_reason == "stop_sequence" and isinstance(stop_sequence, str) and stop_sequence:
        return Constants.STOP_STOP_SEQUENCE, stop_sequence
    return Constants.STOP_END_TURN, None


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


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
