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
    ClaudeContentBlockDocument,
    ClaudeContentBlockImage,
    ClaudeContentBlockText,
]


def build_responses_request(
    claude_request: ClaudeMessagesRequestModel,
    model_mapper: ModelMapperLike,
    *,
    max_tokens_limit: int,
    timeout: int,
    api_key: str,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_supported_claude_fields_for_responses(claude_request)

    responses_request: Dict[str, Any] = {
        "api_key": api_key,
        "input": convert_claude_messages_to_responses_input(claude_request.messages),
        "max_output_tokens": min(claude_request.max_tokens, max_tokens_limit),
        "model": model_mapper.map_claude_to_openai(claude_request.model),
        "stream": claude_request.stream,
        "timeout": timeout,
    }

    instructions = convert_system_content(claude_request.system)
    if instructions:
        responses_request["instructions"] = instructions

    if claude_request.temperature is not None:
        responses_request["temperature"] = claude_request.temperature
    if claude_request.top_p is not None:
        responses_request["top_p"] = claude_request.top_p
    if claude_request.service_tier:
        responses_request["service_tier"] = claude_request.service_tier

    if claude_request.tools:
        responses_request["tools"] = [
            convert_function_tool_definition(tool) for tool in claude_request.tools
        ]

    if claude_request.tool_choice:
        tool_choice_value, parallel_tool_calls = convert_tool_choice_for_responses(
            claude_request.tool_choice
        )
        responses_request["tool_choice"] = tool_choice_value
        if parallel_tool_calls is not None:
            responses_request["parallel_tool_calls"] = parallel_tool_calls

    reasoning = resolve_reasoning_config(
        output_config=claude_request.output_config,
        thinking=claude_request.thinking,
    )
    if reasoning is not None:
        responses_request["reasoning"] = reasoning

    if api_base:
        responses_request["api_base"] = api_base

    return responses_request


def _validate_supported_claude_fields_for_responses(
    claude_request: ClaudeMessagesRequestModel,
) -> None:
    if claude_request.stop_sequences is not None:
        raise ValueError(
            "Field 'stop_sequences' is not supported with the OpenAI Responses API path."
        )
    if claude_request.top_k is not None:
        raise ValueError("Field 'top_k' is not supported with the OpenAI Responses API path.")
    if claude_request.inference_geo is not None:
        raise ValueError(
            "Field 'inference_geo' is not supported with the OpenAI Responses API path."
        )
    if claude_request.context_management is not None:
        raise ValueError(
            "Field 'context_management' is not supported with the OpenAI Responses API path."
        )

    for index, tool in enumerate(claude_request.tools or []):
        if isinstance(tool, ClaudeWebSearchTool):
            raise ValueError(
                f"Field 'tools[{index}]' uses native web_search, which is not supported."
            )


def convert_claude_messages_to_responses_input(
    messages: Sequence[ClaudeMessage],
) -> List[Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        if message.role == Constants.ROLE_USER:
            input_items.extend(convert_claude_user_message_to_responses_items(message))
            continue
        if message.role == Constants.ROLE_ASSISTANT:
            input_items.extend(convert_claude_assistant_message_to_responses_items(message))
            continue
        raise ValueError(f"Unsupported role '{message.role}' in Claude message history")
    return input_items


def convert_claude_user_message_to_responses_items(message: ClaudeMessage) -> List[Dict[str, Any]]:
    if isinstance(message.content, str):
        return [
            responses_message_item(
                Constants.ROLE_USER,
                [responses_text_part(Constants.ROLE_USER, message.content)],
            )
        ]

    output_items: List[Dict[str, Any]] = []
    pending_parts: List[Dict[str, Any]] = []

    for block in message.content:
        if isinstance(block, ClaudeContentBlockToolResult):
            if pending_parts:
                output_items.append(responses_message_item(Constants.ROLE_USER, pending_parts))
                pending_parts = []
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": parse_tool_result_content(block.content),
                }
            )
            continue

        pending_parts.extend(convert_user_block_to_responses_parts(_as_user_content_block(block)))

    if pending_parts or not output_items:
        output_items.append(responses_message_item(Constants.ROLE_USER, pending_parts))

    return output_items


def convert_claude_assistant_message_to_responses_items(
    message: ClaudeMessage,
) -> List[Dict[str, Any]]:
    if isinstance(message.content, str):
        return [
            responses_message_item(
                Constants.ROLE_ASSISTANT,
                [responses_text_part(Constants.ROLE_ASSISTANT, message.content)],
            )
        ]

    output_items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    def flush_assistant_text() -> None:
        if not text_parts:
            return
        output_items.append(
            responses_message_item(
                Constants.ROLE_ASSISTANT,
                [responses_text_part(Constants.ROLE_ASSISTANT, "".join(text_parts))],
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


def responses_message_item(role: str, content_parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_parts = [part for part in content_parts if isinstance(part, dict)]
    if not normalized_parts:
        normalized_parts = [responses_text_part(role, "")]
    return {
        "type": "message",
        "role": role,
        "content": normalized_parts,
    }


def responses_text_part(role: str, text: str) -> Dict[str, str]:
    part_type = "output_text" if role == Constants.ROLE_ASSISTANT else "input_text"
    return {"type": part_type, "text": text}


def convert_user_block_to_responses_parts(block: UserContentBlock) -> List[Dict[str, Any]]:
    if isinstance(block, ClaudeContentBlockText):
        return [{"type": "input_text", "text": block.text}]

    if isinstance(block, ClaudeContentBlockImage):
        return [convert_image_block_to_responses_input(block)]

    if isinstance(block, ClaudeContentBlockDocument):
        return [{"type": "input_text", "text": convert_document_block_to_text(block)}]

    raise ValueError(f"Unsupported user content block type for conversion: {block.type}")


def convert_tool_choice_for_responses(tool_choice: Any) -> Tuple[Any, Optional[bool]]:
    parallel_tool_calls: Optional[bool] = None

    if isinstance(tool_choice, ClaudeToolChoiceAuto):
        tool_choice_value: Any = "auto"
    elif isinstance(tool_choice, ClaudeToolChoiceAny):
        tool_choice_value = "required"
    elif isinstance(tool_choice, ClaudeToolChoiceNone):
        tool_choice_value = "none"
    elif isinstance(tool_choice, ClaudeToolChoiceTool):
        tool_choice_value = {"type": Constants.TOOL_FUNCTION, "name": tool_choice.name}
    else:
        raise ValueError("Unsupported tool_choice value for conversion")

    if getattr(tool_choice, "disable_parallel_tool_use", None) is not None:
        parallel_tool_calls = not bool(tool_choice.disable_parallel_tool_use)

    return tool_choice_value, parallel_tool_calls


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
                item_type = item.get("type")
                if item_type == Constants.CONTENT_TEXT and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item_type == Constants.CONTENT_DOCUMENT:
                    parts.append(convert_document_dict_to_text(item))
                elif item_type == Constants.CONTENT_IMAGE:
                    parts.append("[image content omitted]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type == Constants.CONTENT_TEXT and isinstance(content.get("text"), str):
            text_value = content.get("text")
            return text_value if isinstance(text_value, str) else ""
        if content_type == Constants.CONTENT_DOCUMENT:
            return convert_document_dict_to_text(content)
        if content_type == Constants.CONTENT_IMAGE:
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
    elif block.source.type == "file":
        text = f"[document file_id: {block.source.file_id}]"
    else:
        text = "[document content omitted]"

    metadata_parts = [part for part in [block.title, block.context] if part]
    if metadata_parts:
        return f"{' | '.join(metadata_parts)}\n{text}"
    return text


def convert_document_dict_to_text(document_block: Dict[str, Any]) -> str:
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
            text = f"[document url: {source_url}]" if isinstance(source_url, str) else "[document]"
        elif source_type == "file":
            file_id = source_data.get("file_id")
            text = f"[document file_id: {file_id}]" if isinstance(file_id, str) else "[document]"

    metadata_parts = [
        part for part in [document_block.get("title"), document_block.get("context")] if part
    ]
    if metadata_parts and text:
        return f"{' | '.join(metadata_parts)}\n{text}"
    if metadata_parts:
        return " | ".join(str(part) for part in metadata_parts)
    return text


def convert_image_block_to_responses_input(block: ClaudeContentBlockImage) -> Dict[str, Any]:
    source = block.source
    if source.type == "base64":
        return {
            "type": "input_image",
            "image_url": f"data:{source.media_type};base64,{source.data}",
        }
    if source.type == "url":
        return {"type": "input_image", "image_url": source.url}
    raise ValueError(f"Unsupported image source type for conversion: {source.type}")


def convert_system_content(
    system_content: Optional[Union[str, Sequence[ClaudeSystemContent]]],
) -> str:
    if system_content is None:
        return ""
    if isinstance(system_content, str):
        return system_content.strip()
    return "\n\n".join(block.text.strip() for block in system_content if block.text.strip())


def convert_function_tool_definition(tool: ClaudeTool) -> Dict[str, Any]:
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
        raise ValueError("Native web_search tool variants are not supported.")

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


def resolve_reasoning_config(
    *,
    output_config: Optional[Dict[str, Any]],
    thinking: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    effort = resolve_reasoning_effort(output_config=output_config, thinking=thinking)
    if effort is None:
        return None
    return {"effort": effort}


def resolve_reasoning_effort(
    *,
    output_config: Optional[Dict[str, Any]],
    thinking: Optional[Dict[str, Any]],
) -> Optional[str]:
    if isinstance(output_config, dict):
        output_effort = map_output_effort_to_openai_reasoning_effort(output_config.get("effort"))
        if output_effort is not None:
            return output_effort

    if not isinstance(thinking, dict):
        return None

    thinking_type = thinking.get("type")
    if thinking_type in {"adaptive", "disabled"}:
        return None
    if thinking_type != "enabled":
        return None

    return map_thinking_budget_to_reasoning_effort(thinking.get("budget_tokens"))


def map_output_effort_to_openai_reasoning_effort(effort: Any) -> Optional[str]:
    if effort == "low":
        return "medium"
    if effort == "medium":
        return "high"
    if effort in {"high", "max"}:
        return "xhigh"
    return None


def map_thinking_budget_to_reasoning_effort(budget_tokens: Any) -> Optional[str]:
    if not isinstance(budget_tokens, int):
        return None
    if budget_tokens < 1024:
        return "medium"
    if budget_tokens < 4096:
        return "high"
    return "xhigh"
