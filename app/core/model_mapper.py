"""Model mapping from Claude model names to OpenAI model names."""


class ModelMapper:
    """Maps Claude model names to OpenAI model names based on configuration."""

    def __init__(self, big_model: str, middle_model: str, small_model: str) -> None:
        self.big_model = big_model
        self.middle_model = middle_model
        self.small_model = small_model

        # Pass-through prefixes (don't map these)
        self.passthrough_prefixes = (
            "gpt-",
            "o1-",
            "o3-",
            "o4-",
            "gpt-5",
            "ep-",
            "doubao-",
            "deepseek-",
        )

    def map_claude_to_openai(self, claude_model: str) -> str:
        """Map Claude model name to OpenAI model name."""
        # Pass-through models
        for prefix in self.passthrough_prefixes:
            if claude_model.startswith(prefix):
                return claude_model

        # Map by substring heuristic
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.small_model
        if "sonnet" in model_lower:
            return self.middle_model
        if "opus" in model_lower:
            return self.big_model

        # Default to big model
        return self.big_model
