from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForwardCompatBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClaudeCacheControl(StrictBaseModel):
    type: Literal["ephemeral"]


class ClaudeContentBlockText(StrictBaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeImageSourceBase64(StrictBaseModel):
    type: Literal["base64"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ClaudeImageSourceURL(StrictBaseModel):
    type: Literal["url"]
    url: str


ClaudeImageSource = Annotated[
    Union[ClaudeImageSourceBase64, ClaudeImageSourceURL],
    Field(discriminator="type"),
]


class ClaudeContentBlockImage(StrictBaseModel):
    type: Literal["image"]
    source: ClaudeImageSource
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeDocumentSourceBase64(StrictBaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class ClaudeDocumentSourceText(StrictBaseModel):
    type: Literal["text"]
    text: str


class ClaudeDocumentSourceURL(StrictBaseModel):
    type: Literal["url"]
    url: str


class ClaudeDocumentSourceFile(StrictBaseModel):
    type: Literal["file"]
    file_id: str


ClaudeDocumentSource = Annotated[
    Union[
        ClaudeDocumentSourceBase64,
        ClaudeDocumentSourceText,
        ClaudeDocumentSourceURL,
        ClaudeDocumentSourceFile,
    ],
    Field(discriminator="type"),
]


class ClaudeContentBlockDocument(StrictBaseModel):
    type: Literal["document"]
    source: ClaudeDocumentSource
    title: Optional[str] = None
    context: Optional[str] = None
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeContentBlockToolUse(StrictBaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[ClaudeCacheControl] = None


ToolResultContentBlock = Annotated[
    Union[ClaudeContentBlockText, ClaudeContentBlockImage, ClaudeContentBlockDocument],
    Field(discriminator="type"),
]


class ClaudeContentBlockToolResult(StrictBaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[ToolResultContentBlock]]
    is_error: Optional[bool] = None
    cache_control: Optional[ClaudeCacheControl] = None


class ClaudeContentBlockThinking(StrictBaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str


class ClaudeContentBlockRedactedThinking(StrictBaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ClaudeSystemContent(StrictBaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[ClaudeCacheControl] = None


ClaudeMessageContentBlock = Annotated[
    Union[
        ClaudeContentBlockText,
        ClaudeContentBlockImage,
        ClaudeContentBlockDocument,
        ClaudeContentBlockToolUse,
        ClaudeContentBlockToolResult,
        ClaudeContentBlockThinking,
        ClaudeContentBlockRedactedThinking,
    ],
    Field(discriminator="type"),
]


class ClaudeMessage(StrictBaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ClaudeMessageContentBlock]]

    @model_validator(mode="after")
    def validate_role_content(self) -> "ClaudeMessage":
        if isinstance(self.content, str):
            return self

        user_allowed = {"text", "image", "document", "tool_result"}
        assistant_allowed = {"text", "tool_use", "thinking", "redacted_thinking"}
        allowed_types = user_allowed if self.role == "user" else assistant_allowed

        for block in self.content:
            if block.type not in allowed_types:
                raise ValueError(
                    f"Content block type '{block.type}' is not allowed for role '{self.role}'"
                )
        return self


class ClaudeFunctionTool(StrictBaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]
    input_examples: Optional[List[Dict[str, Any]]] = None


class ClaudeWebSearchUserLocation(StrictBaseModel):
    type: Literal["approximate"]
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None


class ClaudeWebSearchTool(StrictBaseModel):
    type: str = Field(pattern=r"^web_search(_\d{8})?$")
    name: Literal["web_search"]
    max_uses: Optional[int] = Field(default=None, ge=1)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[ClaudeWebSearchUserLocation] = None
    cache_control: Optional[ClaudeCacheControl] = None

    @model_validator(mode="after")
    def validate_domain_filters(self) -> "ClaudeWebSearchTool":
        if self.allowed_domains and self.blocked_domains:
            raise ValueError("web_search tool cannot set both allowed_domains and blocked_domains")
        return self


ClaudeTool = Union[ClaudeFunctionTool, ClaudeWebSearchTool]


class ClaudeMetadata(StrictBaseModel):
    user_id: Optional[str] = None


class ClaudeOutputConfig(StrictBaseModel):
    effort: Optional[Literal["low", "medium", "high", "max"]] = None


class ClaudeThinkingTurnsKeep(StrictBaseModel):
    type: Literal["thinking_turns"]
    value: int = Field(ge=1)


class ClaudeContextEditClearThinking(StrictBaseModel):
    type: Literal["clear_thinking_20251015"]
    keep: Optional[Union[Literal["all"], ClaudeThinkingTurnsKeep]] = None


class ClaudeContextEditClearToolUses(StrictBaseModel):
    type: Literal["clear_tool_uses_20250919"]


ClaudeContextEdit = Annotated[
    Union[ClaudeContextEditClearThinking, ClaudeContextEditClearToolUses],
    Field(discriminator="type"),
]


class ClaudeContextManagement(StrictBaseModel):
    edits: List[ClaudeContextEdit] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_edit_order(self) -> "ClaudeContextManagement":
        if len(self.edits) > 1 and self.edits[0].type != "clear_thinking_20251015":
            raise ValueError(
                "clear_thinking_20251015 must be the first context_management edit when combining edits"
            )
        return self


class ClaudeThinkingConfig(StrictBaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_budget_tokens(self) -> "ClaudeThinkingConfig":
        if self.type != "enabled" and self.budget_tokens is not None:
            raise ValueError("budget_tokens is only valid when thinking.type is 'enabled'")
        return self


class ClaudeToolChoiceAuto(StrictBaseModel):
    type: Literal["auto"]
    disable_parallel_tool_use: Optional[bool] = None


class ClaudeToolChoiceAny(StrictBaseModel):
    type: Literal["any"]
    disable_parallel_tool_use: Optional[bool] = None


class ClaudeToolChoiceTool(StrictBaseModel):
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: Optional[bool] = None


class ClaudeToolChoiceNone(StrictBaseModel):
    type: Literal["none"]


ClaudeToolChoice = Annotated[
    Union[ClaudeToolChoiceAuto, ClaudeToolChoiceAny, ClaudeToolChoiceTool, ClaudeToolChoiceNone],
    Field(discriminator="type"),
]


class _ClaudeMessagesRequestFields(BaseModel):
    model: str
    max_tokens: int = Field(ge=1)
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    output_config: Optional[ClaudeOutputConfig] = None
    metadata: Optional[ClaudeMetadata] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[ClaudeToolChoice] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    service_tier: Optional[str] = None
    inference_geo: Optional[Dict[str, Any]] = None
    context_management: Optional[ClaudeContextManagement] = None

    @model_validator(mode="after")
    def validate_tool_choice(self) -> "_ClaudeMessagesRequestFields":
        if self.tool_choice and self.tool_choice.type != "none" and not self.tools:
            raise ValueError("tool_choice requires tools")
        if self.thinking and self.tool_choice and self.tool_choice.type not in {"auto", "none"}:
            raise ValueError("thinking supports only tool_choice 'auto' or 'none'")
        return self


class ClaudeMessagesRequest(StrictBaseModel, _ClaudeMessagesRequestFields):
    pass


class ClaudeMessagesRequestForwardCompat(ForwardCompatBaseModel, _ClaudeMessagesRequestFields):
    pass


class _ClaudeTokenCountRequestFields(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    output_config: Optional[ClaudeOutputConfig] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[ClaudeToolChoice] = None
    context_management: Optional[ClaudeContextManagement] = None


class ClaudeTokenCountRequest(StrictBaseModel, _ClaudeTokenCountRequestFields):
    pass


class ClaudeTokenCountRequestForwardCompat(ForwardCompatBaseModel, _ClaudeTokenCountRequestFields):
    pass


ClaudeMessagesRequestModel = Union[ClaudeMessagesRequest, ClaudeMessagesRequestForwardCompat]
ClaudeTokenCountRequestModel = Union[
    ClaudeTokenCountRequest,
    ClaudeTokenCountRequestForwardCompat,
]


def parse_claude_messages_request(
    payload: Dict[str, Any], allow_unknown_fields: bool
) -> ClaudeMessagesRequestModel:
    model: Type[ClaudeMessagesRequestModel]
    if allow_unknown_fields:
        model = ClaudeMessagesRequestForwardCompat
    else:
        model = ClaudeMessagesRequest
    return model.model_validate(payload)


def parse_claude_token_count_request(
    payload: Dict[str, Any], allow_unknown_fields: bool
) -> ClaudeTokenCountRequestModel:
    model: Type[ClaudeTokenCountRequestModel]
    if allow_unknown_fields:
        model = ClaudeTokenCountRequestForwardCompat
    else:
        model = ClaudeTokenCountRequest
    return model.model_validate(payload)
