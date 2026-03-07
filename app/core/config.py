import os
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    def __init__(self) -> None:
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8000"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.uvicorn_workers = int(os.environ.get("UVICORN_WORKERS", "1"))
        if self.uvicorn_workers < 1:
            raise ValueError("UVICORN_WORKERS must be >= 1")

        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))

        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

    def validate_client_api_key(self, client_api_key: Optional[str]) -> bool:
        """Validate client's Anthropic API key."""
        if not self.anthropic_api_key:
            return True
        return client_api_key == self.anthropic_api_key

    def ensure_openai_api_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")
        return self.openai_api_key


config = Config()
