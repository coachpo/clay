from typing import Protocol

from app.core.config import config


class ConfigLike(Protocol):
    small_model: str
    middle_model: str
    big_model: str


class ModelManager:
    def __init__(self, config_obj: ConfigLike) -> None:
        self.config = config_obj

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to configured OpenAI models."""
        if claude_model.startswith(("gpt-", "o1-", "o3-", "o4-", "gpt-5")):
            return claude_model

        if claude_model.startswith(("ep-", "doubao-", "deepseek-")):
            return claude_model

        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return str(self.config.small_model)
        if "sonnet" in model_lower:
            return str(self.config.middle_model)
        if "opus" in model_lower:
            return str(self.config.big_model)

        return str(self.config.big_model)


model_manager = ModelManager(config)
