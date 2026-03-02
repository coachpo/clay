from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from src.core.config import config
from src.core.constants import Constants
from src.models.claude import (
    ClaudeContentBlockDocument,
    ClaudeContentBlockImage,
    ClaudeContentBlockRedactedThinking,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessageContentBlock,
    ClaudeMessagesRequestModel,
    ClaudeSystemContent,
    ClaudeTool,
)

logger = logging.getLogger(__name__)


class ModelMapper(Protocol):
    def map_claude_model_to_openai(self, claude_model: str) -> str: ...


UserContentBlock = Union[
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockDocument,
]


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequestModel,
    model_manager: ModelMapper,
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI chat.completions format."""
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)
    openai_messages: List[Dict[str, Any]] = []

    if claude_request.system:
        system_text = _convert_system_content(claude_request.system)
        if system_text:
            openai_messages.append({"role": Constants.ROLE_SYSTEM, "content": system_text})

    for message in claude_request.messages:
        if message.role == Constants.ROLE_USER:
            openai_messages.extend(convert_claude_user_message(message))
        elif message.role == Constants.ROLE_ASSISTANT:
            openai_messages.append(convert_claude_assistant_message(message))

    max_tokens = min(
        max(claude_request.max_tokens, config.min_tokens_limit),
        config.max_tokens_limit,
    )

    openai_request: Dict[str, Any] = {
        "model": openai_model,
        "messages": openai_messages,
        "stream": claude_request.stream,
    }

    if _uses_max_completion_tokens(openai_model):
        openai_request["max_completion_tokens"] = max_tokens
    else:
        openai_request["max_tokens"] = max_tokens

    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None and not _uses_max_completion_tokens(openai_model):
        openai_request["top_p"] = claude_request.top_p
    if claude_request.temperature is not None and not _uses_max_completion_tokens(openai_model):
        openai_request["temperature"] = claude_request.temperature
    if claude_request.metadata is not None:
        openai_request["metadata"] = claude_request.metadata.model_dump(exclude_none=True)
    if claude_request.service_tier is not None:
        openai_request["service_tier"] = claude_request.service_tier

    supports_extension_passthrough = _supports_extension_passthrough(config.openai_base_url)
    extra_body: Dict[str, Any] = {}

    if claude_request.top_k is not None:
        if supports_extension_passthrough:
            extra_body["top_k"] = claude_request.top_k
        else:
            logger.info("Ignoring top_k because target provider does not support it reliably")

    if claude_request.thinking is not None:
        thinking_payload = claude_request.thinking.model_dump(exclude_none=True)
        if supports_extension_passthrough:
            extra_body["thinking"] = thinking_payload
        if (
            claude_request.thinking.type == "enabled"
            and _uses_max_completion_tokens(openai_model)
            and "reasoning_effort" not in openai_request
        ):
            openai_request["reasoning_effort"] = _map_thinking_budget_to_reasoning_effort(
                claude_request.thinking.budget_tokens
            )

    if claude_request.inference_geo is not None:
        if supports_extension_passthrough:
            extra_body["inference_geo"] = claude_request.inference_geo
        else:
            logger.info(
                "Ignoring inference_geo because target provider does not support it reliably"
            )

    if extra_body:
        existing_extra_body = openai_request.get("extra_body")
        if isinstance(existing_extra_body, dict):
            existing_extra_body.update(extra_body)
        else:
            openai_request["extra_body"] = extra_body

    if claude_request.tools:
        openai_request["tools"] = [_convert_tool_definition(tool) for tool in claude_request.tools]

    if claude_request.tool_choice:
        tool_choice = claude_request.tool_choice
        if tool_choice.type == "auto":
            openai_request["tool_choice"] = "auto"
            if tool_choice.disable_parallel_tool_use is not None:
                openai_request["parallel_tool_calls"] = not tool_choice.disable_parallel_tool_use
        elif tool_choice.type == "any":
            openai_request["tool_choice"] = "required"
            if tool_choice.disable_parallel_tool_use is not None:
                openai_request["parallel_tool_calls"] = not tool_choice.disable_parallel_tool_use
        elif tool_choice.type == "tool":
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": tool_choice.name},
            }
            if tool_choice.disable_parallel_tool_use is not None:
                openai_request["parallel_tool_calls"] = not tool_choice.disable_parallel_tool_use
        elif tool_choice.type == "none":
            openai_request["tool_choice"] = "none"

    logger.debug(
        "Converted Claude request to OpenAI format: %s",
        json.dumps(openai_request, ensure_ascii=False),
    )
    return openai_request


def convert_claude_user_message(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert one Claude user message into one or more OpenAI messages."""
    if isinstance(msg.content, str):
        return [{"role": Constants.ROLE_USER, "content": msg.content}]

    converted_messages: List[Dict[str, Any]] = []
    pending_user_blocks: List[UserContentBlock] = []

    for block in msg.content:
        if isinstance(block, ClaudeContentBlockToolResult):
            if pending_user_blocks:
                converted_messages.append(
                    {
                        "role": Constants.ROLE_USER,
                        "content": _convert_user_content_blocks(pending_user_blocks),
                    }
                )
                pending_user_blocks = []

            tool_text = parse_tool_result_content(block.content)
            if block.is_error:
                tool_text = f"__tool_error__\n{tool_text}"

            converted_messages.append(
                {
                    "role": Constants.ROLE_TOOL,
                    "tool_call_id": block.tool_use_id,
                    "content": tool_text,
                }
            )
            continue

        pending_user_blocks.append(_as_user_content_block(block))

    if pending_user_blocks or not converted_messages:
        converted_messages.append(
            {
                "role": Constants.ROLE_USER,
                "content": _convert_user_content_blocks(pending_user_blocks),
            }
        )

    return converted_messages


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, ClaudeContentBlockText):
            text_parts.append(block.text)
            continue

        if isinstance(block, ClaudeContentBlockToolUse):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )
            continue

        if isinstance(block, ClaudeContentBlockThinking):
            text_parts.append(block.thinking)
            continue

        if isinstance(block, ClaudeContentBlockRedactedThinking):
            text_parts.append("[redacted thinking]")
            continue

        raise ValueError(f"Unsupported assistant content block type for conversion: {block.type}")

    openai_message: Dict[str, Any] = {"role": Constants.ROLE_ASSISTANT}
    openai_message["content"] = "".join(text_parts) if text_parts else None
    if tool_calls:
        openai_message["tool_calls"] = tool_calls
    return openai_message


def parse_tool_result_content(content: Any) -> str:
    """Normalize Anthropic tool_result content into OpenAI tool message text."""
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
                parts.append(_convert_document_block_to_text(item))
            elif isinstance(item, ClaudeContentBlockImage):
                parts.append("[image content omitted]")
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == Constants.CONTENT_TEXT:
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                elif item_type == Constants.CONTENT_DOCUMENT:
                    parts.append(_convert_document_dict_to_text(item))
                elif item_type == Constants.CONTENT_IMAGE:
                    parts.append("[image content omitted]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type == Constants.CONTENT_TEXT:
            text_value = content.get("text")
            return text_value if isinstance(text_value, str) else ""
        if content_type == Constants.CONTENT_DOCUMENT:
            return _convert_document_dict_to_text(content)
        if content_type == Constants.CONTENT_IMAGE:
            return "[image content omitted]"
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _convert_user_content_blocks(blocks: Sequence[UserContentBlock]) -> Any:
    if not blocks:
        return ""

    openai_content: List[Dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, ClaudeContentBlockText):
            openai_content.append({"type": "text", "text": block.text})
        elif isinstance(block, ClaudeContentBlockImage):
            openai_content.append(_convert_image_block_to_openai_content(block))
        elif isinstance(block, ClaudeContentBlockDocument):
            openai_content.append(
                {
                    "type": "text",
                    "text": _convert_document_block_to_text(block),
                }
            )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return openai_content[0]["text"]
    return openai_content


def _convert_document_block_to_text(block: ClaudeContentBlockDocument) -> str:
    if block.source.type == "text":
        text = block.source.text
    elif block.source.type == "base64":
        text = f"[document {block.source.media_type} base64 data omitted]"
    elif block.source.type == "url":
        text = f"[document url: {block.source.url}]"
    elif block.source.type == "file":
        text = f"[document file_id: {block.source.file_id}]"
    else:
        text = "[document content omitted]"

    metadata_parts = [part for part in [block.title, block.context] if part]
    if metadata_parts:
        metadata = " | ".join(metadata_parts)
        return f"{metadata}\n{text}"
    return text


def _convert_document_dict_to_text(document_block: Dict[str, Any]) -> str:
    source_data = document_block.get("source")
    text = ""

    if isinstance(source_data, dict):
        source_type = source_data.get("type")
        if source_type == "text":
            text_value = source_data.get("text")
            text = text_value if isinstance(text_value, str) else ""
        elif source_type == "base64":
            media_type = source_data.get("media_type")
            media_value = media_type if isinstance(media_type, str) else "application/octet-stream"
            text = f"[document {media_value} base64 data omitted]"
        elif source_type == "url":
            source_url = source_data.get("url")
            if isinstance(source_url, str) and source_url:
                text = f"[document url: {source_url}]"
            else:
                text = "[document url]"
        elif source_type == "file":
            file_id = source_data.get("file_id")
            if isinstance(file_id, str) and file_id:
                text = f"[document file_id: {file_id}]"
            else:
                text = "[document file]"

    metadata_parts_raw = [document_block.get("title"), document_block.get("context")]
    metadata_parts = [part for part in metadata_parts_raw if isinstance(part, str) and part]

    if metadata_parts and text:
        return f"{' | '.join(metadata_parts)}\n{text}"
    if metadata_parts:
        return " | ".join(metadata_parts)
    return text


def _convert_image_block_to_openai_content(
    block: ClaudeContentBlockImage,
) -> Dict[str, Any]:
    source = block.source
    if source.type == "base64":
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{source.media_type};base64,{source.data}"},
        }
    if source.type == "url":
        return {
            "type": "image_url",
            "image_url": {"url": source.url},
        }
    raise ValueError(f"Unsupported image source type for conversion: {source.type}")


def _convert_system_content(
    system_content: Union[str, Sequence[ClaudeSystemContent]],
) -> str:
    if isinstance(system_content, str):
        return system_content.strip()

    return "\n\n".join(
        block.text.strip()
        for block in system_content
        if block.type == Constants.CONTENT_TEXT and block.text.strip()
    )


def _convert_tool_definition(tool: ClaudeTool) -> Dict[str, Any]:
    function_payload: Dict[str, Any] = {
        "name": tool.name,
        "parameters": tool.input_schema,
    }
    if tool.description:
        function_payload["description"] = tool.description

    return {
        "type": Constants.TOOL_FUNCTION,
        Constants.TOOL_FUNCTION: function_payload,
    }


def _as_user_content_block(block: ClaudeMessageContentBlock) -> UserContentBlock:
    if isinstance(
        block,
        (
            ClaudeContentBlockText,
            ClaudeContentBlockImage,
            ClaudeContentBlockDocument,
        ),
    ):
        return block

    raise ValueError(f"Unsupported user content block type for conversion: {block.type}")


def _uses_max_completion_tokens(openai_model: str) -> bool:
    model_name = openai_model.lower()
    reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
    return model_name.startswith(reasoning_prefixes)


def _supports_extension_passthrough(base_url: str) -> bool:
    normalized = base_url.lower()
    return "api.openai.com" not in normalized and "openai.azure.com" not in normalized


def _map_thinking_budget_to_reasoning_effort(budget_tokens: Optional[int]) -> str:
    if budget_tokens is None:
        return "medium"
    if budget_tokens < 1024:
        return "low"
    if budget_tokens < 4096:
        return "medium"
    return "high"
