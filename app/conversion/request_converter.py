from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from app.core.constants import Constants
from app.models.claude import (
    ClaudeContentBlockDocument,
    ClaudeContentBlockImage,
    ClaudeContentBlockRedactedThinking,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeFunctionTool,
    ClaudeMessage,
    ClaudeMessageContentBlock,
    ClaudeMessagesRequestModel,
    ClaudeSystemContent,
    ClaudeTool,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceAuto,
    ClaudeToolChoiceNone,
    ClaudeToolChoiceTool,
    ClaudeWebSearchTool,
)


class ModelMapperLike(Protocol):
    def map_claude_to_openai(self, claude_model: str) -> str: ...


UserContentBlock = Union[
    ClaudeContentBlockDocument, ClaudeContentBlockImage, ClaudeContentBlockText
]


def build_completion_request(
    claude_request: ClaudeMessagesRequestModel,
    model_mapper: ModelMapperLike,
    *,
    max_tokens_limit: int,
    request_id: str,
    timeout: int,
    api_key: str,
) -> Dict[str, Any]:
    completion_request: Dict[str, Any] = {
        "api_key": api_key,
        "max_tokens": min(claude_request.max_tokens, max_tokens_limit),
        "messages": convert_claude_messages_to_openai(claude_request),
        "metadata": {"request_id": request_id},
        "model": model_mapper.map_claude_to_openai(claude_request.model),
        "stream": claude_request.stream,
        "timeout": timeout,
    }

    if claude_request.temperature is not None:
        completion_request["temperature"] = claude_request.temperature
    if claude_request.top_p is not None:
        completion_request["top_p"] = claude_request.top_p
    if claude_request.stop_sequences:
        completion_request["stop"] = claude_request.stop_sequences
    if claude_request.service_tier:
        completion_request["service_tier"] = claude_request.service_tier
    if claude_request.metadata:
        completion_request["metadata"].update(claude_request.metadata)

    if claude_request.tools:
        completion_request["tools"] = [
            convert_tool_definition(tool) for tool in claude_request.tools
        ]

    if claude_request.tool_choice:
        tool_choice, parallel_tool_calls = convert_tool_choice(claude_request.tool_choice)
        completion_request["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            completion_request["parallel_tool_calls"] = parallel_tool_calls

    return completion_request


def convert_claude_messages_to_openai(
    claude_request: ClaudeMessagesRequestModel,
) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    system_text = convert_system_content(claude_request.system)
    if system_text:
        openai_messages.append({"role": Constants.ROLE_SYSTEM, "content": system_text})

    for message in claude_request.messages:
        if message.role == Constants.ROLE_USER:
            openai_messages.extend(convert_user_message(message))
        else:
            openai_messages.append(convert_assistant_message(message))

    return openai_messages


def convert_user_message(message: ClaudeMessage) -> List[Dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"role": Constants.ROLE_USER, "content": message.content}]

    messages: List[Dict[str, Any]] = []
    pending_parts: List[Dict[str, Any]] = []

    def flush_user_parts() -> None:
        if not pending_parts:
            return
        messages.append({"role": Constants.ROLE_USER, "content": list(pending_parts)})
        pending_parts.clear()

    for block in message.content:
        if isinstance(block, ClaudeContentBlockToolResult):
            flush_user_parts()
            messages.append(
                {
                    "role": Constants.ROLE_TOOL,
                    "tool_call_id": block.tool_use_id,
                    "content": parse_tool_result_content(block.content),
                }
            )
            continue

        pending_parts.extend(convert_user_block(_as_user_content_block(block)))

    flush_user_parts()
    return messages


def convert_assistant_message(message: ClaudeMessage) -> Dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": message.content}

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ClaudeContentBlockText):
            text_parts.append(block.text)
            continue
        if isinstance(block, ClaudeContentBlockThinking):
            text_parts.append(block.thinking)
            continue
        if isinstance(block, ClaudeContentBlockRedactedThinking):
            text_parts.append("[redacted thinking]")
            continue
        if isinstance(block, ClaudeContentBlockToolUse):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    "function": {
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                        "name": block.name,
                    },
                }
            )

    payload: Dict[str, Any] = {
        "role": Constants.ROLE_ASSISTANT,
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def convert_user_block(block: UserContentBlock) -> List[Dict[str, Any]]:
    if isinstance(block, ClaudeContentBlockText):
        return [{"type": Constants.CONTENT_TEXT, "text": block.text}]

    if isinstance(block, ClaudeContentBlockDocument):
        return [{"type": Constants.CONTENT_TEXT, "text": convert_document_block_to_text(block)}]

    if isinstance(block, ClaudeContentBlockImage):
        if block.source.type == "base64":
            return [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.source.media_type};base64,{block.source.data}",
                    },
                }
            ]
        return [{"type": "image_url", "image_url": {"url": block.source.url}}]

    raise ValueError(f"Unsupported user content block type for conversion: {block.type}")


def parse_tool_result_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, ClaudeContentBlockText):
                parts.append(item.text)
            elif isinstance(item, ClaudeContentBlockDocument):
                parts.append(convert_document_block_to_text(item))
            elif isinstance(item, ClaudeContentBlockImage):
                parts.append("[image content omitted]")
            elif isinstance(item, dict):
                if item.get("type") == Constants.CONTENT_TEXT and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == Constants.CONTENT_DOCUMENT:
                    parts.append(convert_document_dict_to_text(item))
                elif item.get("type") == Constants.CONTENT_IMAGE:
                    parts.append("[image content omitted]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT and isinstance(content.get("text"), str):
            text_value = content.get("text")
            return text_value if isinstance(text_value, str) else ""
        if content.get("type") == Constants.CONTENT_DOCUMENT:
            return convert_document_dict_to_text(content)
        if content.get("type") == Constants.CONTENT_IMAGE:
            return "[image content omitted]"
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def convert_document_block_to_text(block: ClaudeContentBlockDocument) -> str:
    if block.source.type == "text":
        text = block.source.text
    elif block.source.type == "base64":
        text = f"[document {block.source.media_type} base64 data omitted]"
    elif block.source.type == "url":
        text = f"[document url: {block.source.url}]"
    else:
        text = f"[document file_id: {block.source.file_id}]"

    metadata = " | ".join(part for part in [block.title, block.context] if part)
    if metadata:
        return f"{metadata}\n{text}"
    return text


def convert_document_dict_to_text(document_block: Dict[str, Any]) -> str:
    source = document_block.get("source")
    text = ""
    if isinstance(source, dict):
        source_type = source.get("type")
        if source_type == "text" and isinstance(source.get("text"), str):
            text = source["text"]
        elif source_type == "base64":
            media_type = source.get("media_type") or "application/octet-stream"
            text = f"[document {media_type} base64 data omitted]"
        elif source_type == "url" and isinstance(source.get("url"), str):
            text = f"[document url: {source['url']}]"
        elif source_type == "file" and isinstance(source.get("file_id"), str):
            text = f"[document file_id: {source['file_id']}]"

    metadata = " | ".join(
        value for value in [document_block.get("title"), document_block.get("context")] if value
    )
    if metadata and text:
        return f"{metadata}\n{text}"
    return metadata or text


def convert_system_content(
    system_content: Optional[Union[str, Sequence[ClaudeSystemContent]]],
) -> str:
    if system_content is None:
        return ""
    if isinstance(system_content, str):
        return system_content.strip()
    return "\n\n".join(block.text.strip() for block in system_content if block.text.strip())


def convert_tool_definition(tool: ClaudeTool) -> Dict[str, Any]:
    if isinstance(tool, ClaudeFunctionTool):
        function: Dict[str, Any] = {
            "name": tool.name,
            "parameters": tool.input_schema,
        }
        if tool.description:
            function["description"] = tool.description
        return {"type": Constants.TOOL_FUNCTION, "function": function}

    if isinstance(tool, ClaudeWebSearchTool):
        raise ValueError("Native web_search tools are not supported for /v1/messages.")

    raise ValueError("Unsupported tool definition for conversion")


def convert_tool_choice(tool_choice: Any) -> Tuple[Any, Optional[bool]]:
    parallel_tool_calls: Optional[bool] = None

    if isinstance(tool_choice, ClaudeToolChoiceAuto):
        tool_choice_value: Any = "auto"
    elif isinstance(tool_choice, ClaudeToolChoiceAny):
        tool_choice_value = "required"
    elif isinstance(tool_choice, ClaudeToolChoiceNone):
        tool_choice_value = "none"
    elif isinstance(tool_choice, ClaudeToolChoiceTool):
        tool_choice_value = {
            "type": Constants.TOOL_FUNCTION,
            "function": {"name": tool_choice.name},
        }
    else:
        raise ValueError("Unsupported tool_choice value for conversion")

    if getattr(tool_choice, "disable_parallel_tool_use", None) is not None:
        parallel_tool_calls = not bool(tool_choice.disable_parallel_tool_use)

    return tool_choice_value, parallel_tool_calls


def _as_user_content_block(block: ClaudeMessageContentBlock) -> UserContentBlock:
    if isinstance(
        block, (ClaudeContentBlockDocument, ClaudeContentBlockImage, ClaudeContentBlockText)
    ):
        return block
    raise ValueError(f"Unsupported user content block type for conversion: {block.type}")
