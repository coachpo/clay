import os
import sys
from typing import List, Optional


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


class Config:
    def __init__(self) -> None:
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")

        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8000"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.uvicorn_workers = int(os.environ.get("UVICORN_WORKERS", "1"))
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        if self.uvicorn_workers < 1:
            raise ValueError("UVICORN_WORKERS must be >= 1")

        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        self.anthropic_default_version = os.environ.get("ANTHROPIC_DEFAULT_VERSION", "2023-06-01")
        supported_versions_raw = os.environ.get(
            "ANTHROPIC_SUPPORTED_VERSIONS", self.anthropic_default_version
        )
        parsed_versions = _parse_csv(supported_versions_raw)
        if not parsed_versions:
            parsed_versions = [self.anthropic_default_version]
        if self.anthropic_default_version not in parsed_versions:
            parsed_versions.append(self.anthropic_default_version)
        self.anthropic_supported_versions = sorted(set(parsed_versions))

        self.anthropic_compatibility_mode = _env_flag("ANTHROPIC_COMPATIBILITY_MODE", False)
        self.anthropic_allow_version_fallback = _env_flag(
            "ANTHROPIC_ALLOW_VERSION_FALLBACK", self.anthropic_compatibility_mode
        )
        self.anthropic_allow_missing_version = _env_flag("ANTHROPIC_ALLOW_MISSING_VERSION", False)
        self.anthropic_allow_unknown_fields = _env_flag(
            "ANTHROPIC_ALLOW_UNKNOWN_FIELDS", self.anthropic_compatibility_mode
        )
        # Optional compatibility knob for providers that expect a wider temperature range.
        # Default is disabled to preserve Anthropic client intent and avoid surprise randomness shifts.
        self.anthropic_temperature_scale_to_openai_x2 = _env_flag(
            "ANTHROPIC_TEMPERATURE_SCALE_TO_OPENAI_X2", False
        )
        self.allow_openai_extension_passthrough = _env_flag(
            "ALLOW_OPENAI_EXTENSION_PASSTHROUGH", False
        )
        self.include_original_anthropic_request = _env_flag(
            "INCLUDE_ORIGINAL_ANTHROPIC_REQUEST", self.anthropic_compatibility_mode
        )
        self.openai_responses_state_mode = (
            os.environ.get("OPENAI_RESPONSES_STATE_MODE", "stateless").strip().lower()
        )
        if self.openai_responses_state_mode not in {"stateless", "provider"}:
            raise ValueError("OPENAI_RESPONSES_STATE_MODE must be either 'stateless' or 'provider'")

    def validate_api_key(self) -> bool:
        """Basic API key validation."""
        if not self.openai_api_key:
            return False
        return self.openai_api_key.startswith("sk-")

    def validate_client_api_key(self, client_api_key: Optional[str]) -> bool:
        """Validate client's Anthropic API key."""
        if not self.anthropic_api_key:
            return True
        return client_api_key == self.anthropic_api_key


try:
    config = Config()
    print(
        "Configuration loaded: API_KEY=%s..., BASE_URL='%s', "
        "ANTHROPIC_DEFAULT_VERSION='%s', OPENAI_RESPONSES_STATE_MODE='%s'"
        % (
            "*" * 20,
            config.openai_base_url,
            config.anthropic_default_version,
            config.openai_responses_state_mode,
        )
    )
except Exception as error:
    print(f"Configuration Error: {error}")
    sys.exit(1)
