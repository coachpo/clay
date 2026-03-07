from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaudeBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClaudeCacheControl(ClaudeBaseModel):
    type: Literal["ephemeral"]


class ClaudeContentBlockText(ClaudeBaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeImageSourceBase64(ClaudeBaseModel):
    type: Literal["base64"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ClaudeImageSourceURL(ClaudeBaseModel):
    type: Literal["url"]
    url: str


ClaudeImageSource = Annotated[
    Union[ClaudeImageSourceBase64, ClaudeImageSourceURL],
    Field(discriminator="type"),
]


class ClaudeContentBlockImage(ClaudeBaseModel):
    type: Literal["image"]
    source: ClaudeImageSource
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeDocumentSourceBase64(ClaudeBaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class ClaudeDocumentSourceFile(ClaudeBaseModel):
    type: Literal["file"]
    file_id: str


class ClaudeDocumentSourceText(ClaudeBaseModel):
    type: Literal["text"]
    text: str


class ClaudeDocumentSourceURL(ClaudeBaseModel):
    type: Literal["url"]
    url: str


ClaudeDocumentSource = Annotated[
    Union[
        ClaudeDocumentSourceBase64,
        ClaudeDocumentSourceFile,
        ClaudeDocumentSourceText,
        ClaudeDocumentSourceURL,
    ],
    Field(discriminator="type"),
]


class ClaudeContentBlockDocument(ClaudeBaseModel):
    type: Literal["document"]
    source: ClaudeDocumentSource
    title: Optional[str] = None
    context: Optional[str] = None
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeContentBlockThinking(ClaudeBaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str = ""


class ClaudeContentBlockRedactedThinking(ClaudeBaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ClaudeContentBlockToolUse(ClaudeBaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[ClaudeCacheControl] = None


ToolResultContentBlock = Annotated[
    Union[ClaudeContentBlockText, ClaudeContentBlockImage, ClaudeContentBlockDocument],
    Field(discriminator="type"),
]


class ClaudeContentBlockToolResult(ClaudeBaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[ToolResultContentBlock]]
    is_error: Optional[bool] = None
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeSystemContent(ClaudeBaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[ClaudeCacheControl] = None


ClaudeMessageContentBlock = Annotated[
    Union[
        ClaudeContentBlockDocument,
        ClaudeContentBlockImage,
        ClaudeContentBlockRedactedThinking,
        ClaudeContentBlockText,
        ClaudeContentBlockThinking,
        ClaudeContentBlockToolResult,
        ClaudeContentBlockToolUse,
    ],
    Field(discriminator="type"),
]


class ClaudeMessage(ClaudeBaseModel):
    role: Literal["assistant", "user"]
    content: Union[str, List[ClaudeMessageContentBlock]]

    @model_validator(mode="after")
    def validate_role_content(self) -> "ClaudeMessage":
        if isinstance(self.content, str):
            return self

        user_allowed = {"document", "image", "text", "tool_result"}
        assistant_allowed = {"redacted_thinking", "text", "thinking", "tool_use"}
        allowed_types = user_allowed if self.role == "user" else assistant_allowed

        for block in self.content:
            if block.type not in allowed_types:
                raise ValueError(
                    f"Content block type '{block.type}' is not allowed for role '{self.role}'"
                )
        return self


class ClaudeFunctionTool(ClaudeBaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ClaudeWebSearchTool(ClaudeBaseModel):
    type: str = Field(pattern=r"^web_search(_\d{8})?$")
    name: Literal["web_search"]


ClaudeTool = Union[ClaudeFunctionTool, ClaudeWebSearchTool]


class ClaudeToolChoiceAuto(ClaudeBaseModel):
    type: Literal["auto"]
    disable_parallel_tool_use: Optional[bool] = None


class ClaudeToolChoiceAny(ClaudeBaseModel):
    type: Literal["any"]
    disable_parallel_tool_use: Optional[bool] = None


class ClaudeToolChoiceNone(ClaudeBaseModel):
    type: Literal["none"]


class ClaudeToolChoiceTool(ClaudeBaseModel):
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: Optional[bool] = None


ClaudeToolChoice = Annotated[
    Union[ClaudeToolChoiceAuto, ClaudeToolChoiceAny, ClaudeToolChoiceNone, ClaudeToolChoiceTool],
    Field(discriminator="type"),
]


class ClaudeMessagesRequest(ClaudeBaseModel):
    model: str
    max_tokens: int = Field(ge=1)
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[ClaudeToolChoice] = None
    metadata: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None
    output_config: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None
    context_management: Optional[Dict[str, Any]] = None
    inference_geo: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_tool_choice(self) -> "ClaudeMessagesRequest":
        if self.tool_choice and self.tool_choice.type != "none" and not self.tools:
            raise ValueError("tool_choice requires tools")
        return self


ClaudeMessagesRequestModel = ClaudeMessagesRequest


def parse_claude_messages_request(payload: Dict[str, Any]) -> ClaudeMessagesRequestModel:
    return ClaudeMessagesRequest.model_validate(payload)
