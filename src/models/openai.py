from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class OpenAIChatMessage(OpenAIBaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAIChatCompletionsRequest(OpenAIBaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)


class OpenAIResponsesRequest(OpenAIBaseModel):
    model: str
    input: Any
    stream: bool = False
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
