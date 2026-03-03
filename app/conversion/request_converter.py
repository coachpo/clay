from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from app.core.config import config
from app.core.constants import Constants
from app.models.claude import (
    ClaudeContentBlockDocument,
    ClaudeContentBlockImage,
    ClaudeContentBlockRedactedThinking,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeContextManagement,
    ClaudeFunctionTool,
    ClaudeMessage,
    ClaudeMessageContentBlock,
    ClaudeMessagesRequestModel,
    ClaudeOutputConfig,
    ClaudeSystemContent,
    ClaudeThinkingConfig,
    ClaudeTool,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceAuto,
    ClaudeToolChoiceNone,
    ClaudeToolChoiceTool,
    ClaudeWebSearchTool,
)

logger = logging.getLogger(__name__)
DEFAULT_OPENAI_COMPACTION_THRESHOLD = 200000


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
    """Backward-compatible alias: convert Claude request into OpenAI Responses payload."""
    return convert_claude_to_responses_request(claude_request, model_manager)


def convert_claude_to_responses_request(
    claude_request: ClaudeMessagesRequestModel,
    model_manager: ModelMapper,
) -> Dict[str, Any]:
    """Convert Claude /v1/messages request into OpenAI Responses API payload."""
    _validate_supported_claude_fields_for_responses(claude_request)

    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)
    max_output_tokens = min(
        max(claude_request.max_tokens, config.min_tokens_limit),
        config.max_tokens_limit,
    )

    responses_request: Dict[str, Any] = {
        "model": openai_model,
        "input": _convert_claude_messages_to_responses_input(claude_request.messages),
        "stream": claude_request.stream,
        "max_output_tokens": max_output_tokens,
        "store": config.openai_responses_state_mode == "provider",
    }

    instructions = _convert_system_content(claude_request.system) if claude_request.system else ""
    if instructions:
        responses_request["instructions"] = instructions

    if claude_request.temperature is not None:
        responses_request["temperature"] = _map_temperature_for_openai(claude_request.temperature)
    if claude_request.top_p is not None:
        responses_request["top_p"] = claude_request.top_p
    if claude_request.metadata is not None:
        responses_request["metadata"] = claude_request.metadata.model_dump(exclude_none=True)
    if claude_request.service_tier is not None:
        responses_request["service_tier"] = claude_request.service_tier

    if claude_request.tools:
        responses_request["tools"] = [
            _convert_function_tool_definition(tool) for tool in claude_request.tools
        ]

    if claude_request.tool_choice:
        tool_choice_value, parallel_tool_calls = _convert_tool_choice_for_responses(
            claude_request.tool_choice
        )
        responses_request["tool_choice"] = tool_choice_value
        if parallel_tool_calls is not None:
            responses_request["parallel_tool_calls"] = parallel_tool_calls

    reasoning_effort = _resolve_reasoning_effort(
        thinking=claude_request.thinking,
        output_config=claude_request.output_config,
    )
    if reasoning_effort is not None:
        responses_request["reasoning"] = {"effort": reasoning_effort}
    if claude_request.context_management is not None:
        responses_request["context_management"] = _convert_context_management_for_responses(
            claude_request.context_management
        )

    logger.debug(
        "Converted Claude request to OpenAI Responses format: %s",
        json.dumps(responses_request, ensure_ascii=False),
    )
    return responses_request


def _validate_supported_claude_fields_for_responses(
    claude_request: ClaudeMessagesRequestModel,
) -> None:
    if claude_request.stop_sequences is not None:
        raise ValueError(
            "Field 'stop_sequences' is not supported for /v1/messages in Responses-only mode."
        )
    if claude_request.top_k is not None:
        raise ValueError("Field 'top_k' is not supported for /v1/messages in Responses-only mode.")
    if claude_request.inference_geo is not None:
        raise ValueError(
            "Field 'inference_geo' is not supported for /v1/messages in Responses-only mode."
        )

    for index, tool in enumerate(claude_request.tools or []):
        if isinstance(tool, ClaudeWebSearchTool):
            raise ValueError(
                "Field 'tools[%d]' uses native web_search variants, which are not supported in "
                "Responses-only mode." % index
            )


def _convert_claude_messages_to_responses_input(
    messages: Sequence[ClaudeMessage],
) -> List[Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        if message.role == Constants.ROLE_USER:
            input_items.extend(_convert_claude_user_message_to_responses_items(message))
            continue
        if message.role == Constants.ROLE_ASSISTANT:
            input_items.extend(_convert_claude_assistant_message_to_responses_items(message))
            continue
        raise ValueError(f"Unsupported role '{message.role}' in Claude message history")
    return input_items


def _convert_claude_user_message_to_responses_items(
    message: ClaudeMessage,
) -> List[Dict[str, Any]]:
    if isinstance(message.content, str):
        return [
            _responses_message_item(
                Constants.ROLE_USER,
                [_responses_text_part(Constants.ROLE_USER, message.content)],
            )
        ]

    output_items: List[Dict[str, Any]] = []
    pending_parts: List[Dict[str, Any]] = []

    for block in message.content:
        if isinstance(block, ClaudeContentBlockToolResult):
            if pending_parts:
                output_items.append(_responses_message_item(Constants.ROLE_USER, pending_parts))
                pending_parts = []
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": parse_tool_result_content(block.content),
                }
            )
            continue

        pending_parts.extend(_convert_user_block_to_responses_parts(_as_user_content_block(block)))

    if pending_parts or not output_items:
        output_items.append(_responses_message_item(Constants.ROLE_USER, pending_parts))

    return output_items


def _convert_claude_assistant_message_to_responses_items(
    message: ClaudeMessage,
) -> List[Dict[str, Any]]:
    if isinstance(message.content, str):
        return [
            _responses_message_item(
                Constants.ROLE_ASSISTANT,
                [_responses_text_part(Constants.ROLE_ASSISTANT, message.content)],
            )
        ]

    output_items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    def flush_assistant_text() -> None:
        if not text_parts:
            return
        output_items.append(
            _responses_message_item(
                Constants.ROLE_ASSISTANT,
                [_responses_text_part(Constants.ROLE_ASSISTANT, "".join(text_parts))],
            )
        )
        text_parts.clear()

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
            flush_assistant_text()
            output_items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input, ensure_ascii=False),
                }
            )
            continue

        raise ValueError(f"Unsupported assistant content block type for conversion: {block.type}")

    flush_assistant_text()
    return output_items


def _responses_message_item(role: str, content_parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_parts = [part for part in content_parts if isinstance(part, dict)]
    if not normalized_parts:
        normalized_parts = [_responses_text_part(role, "")]
    return {
        "type": "message",
        "role": role,
        "content": normalized_parts,
    }


def _responses_text_part(role: str, text: str) -> Dict[str, str]:
    part_type = "output_text" if role == Constants.ROLE_ASSISTANT else "input_text"
    return {"type": part_type, "text": text}


def _convert_user_block_to_responses_parts(block: UserContentBlock) -> List[Dict[str, Any]]:
    if isinstance(block, ClaudeContentBlockText):
        return [{"type": "input_text", "text": block.text}]

    if isinstance(block, ClaudeContentBlockImage):
        return [_convert_image_block_to_responses_input(block)]

    if isinstance(block, ClaudeContentBlockDocument):
        return [{"type": "input_text", "text": _convert_document_block_to_text(block)}]

    raise ValueError(f"Unsupported user content block type for conversion: {block.type}")


def _convert_tool_choice_for_responses(tool_choice: Any) -> tuple[Any, Optional[bool]]:
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
            "name": tool_choice.name,
        }
    else:
        raise ValueError("Unsupported tool_choice value for conversion")

    if getattr(tool_choice, "disable_parallel_tool_use", None) is not None:
        parallel_tool_calls = not bool(tool_choice.disable_parallel_tool_use)

    return tool_choice_value, parallel_tool_calls


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


def _convert_image_block_to_responses_input(
    block: ClaudeContentBlockImage,
) -> Dict[str, Any]:
    source = block.source
    if source.type == "base64":
        return {
            "type": "input_image",
            "image_url": f"data:{source.media_type};base64,{source.data}",
        }
    if source.type == "url":
        return {
            "type": "input_image",
            "image_url": source.url,
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


def _convert_function_tool_definition(tool: ClaudeTool) -> Dict[str, Any]:
    if isinstance(tool, ClaudeFunctionTool):
        responses_tool: Dict[str, Any] = {
            "type": Constants.TOOL_FUNCTION,
            "name": tool.name,
            "parameters": tool.input_schema,
        }
        if tool.description:
            responses_tool["description"] = tool.description
        return responses_tool

    if isinstance(tool, ClaudeWebSearchTool):
        raise ValueError(
            "Native web_search tool variants are not supported for /v1/messages in Responses-only mode."
        )

    raise ValueError("Unsupported tool definition for conversion")


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


def _map_thinking_budget_to_reasoning_effort(budget_tokens: Optional[int]) -> Optional[str]:
    if budget_tokens is None:
        return None
    if budget_tokens < 1024:
        return "medium"
    if budget_tokens < 4096:
        return "high"
    return "xhigh"


def _resolve_reasoning_effort(
    thinking: Optional[ClaudeThinkingConfig],
    output_config: Optional[ClaudeOutputConfig],
) -> Optional[str]:
    if output_config is not None:
        output_effort = _map_output_effort_to_openai_reasoning_effort(output_config.effort)
        if output_effort is not None:
            return output_effort

    if thinking is None:
        return None

    if thinking.type == "disabled":
        return None
    if thinking.type == "adaptive":
        return None
    return _map_thinking_budget_to_reasoning_effort(thinking.budget_tokens)


def _map_output_effort_to_openai_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    if effort is None:
        return None

    if effort == "low":
        return "medium"
    if effort == "medium":
        return "high"
    if effort in {"high", "max"}:
        return "xhigh"
    return None


def _map_temperature_for_openai(anthropic_temperature: float) -> float:
    # Keep direct passthrough by default for Anthropic compatibility.
    # x2 mapping is opt-in for specific OpenAI-compatible upstreams that prefer 0..2 tuning.
    if not config.anthropic_temperature_scale_to_openai_x2:
        return anthropic_temperature
    return anthropic_temperature * 2


def _convert_context_management_for_responses(
    context_management: ClaudeContextManagement,
) -> List[Dict[str, Any]]:
    # Anthropic context edits have no one-to-one Responses equivalent.
    # OpenAI currently supports compaction entries, so map any request to compaction.
    if not context_management.edits:
        return []

    return [
        {
            "type": "compaction",
            "compact_threshold": DEFAULT_OPENAI_COMPACTION_THRESHOLD,
        }
    ]
