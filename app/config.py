"""Configuration management for the multi-agent PRD generator.

This module loads and validates environment variables and provides
a centralized configuration object.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        """Initialize configuration by loading environment variables."""
        # Load .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        # OpenAI Configuration
        self.openai_api_key: str = self._get_required_env("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")

        # Logging Configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # Rate Limiting Configuration
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay: int = int(os.getenv("RETRY_DELAY", "1"))

        # Directory Configuration
        self.output_dir: Path = Path(os.getenv("OUTPUT_DIR", "data/runs"))
        self.log_dir: Path = Path(os.getenv("LOG_DIR", "data/logs"))

        # Create necessary directories
        self._ensure_directories()

    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable.

        Args:
            key: The environment variable name

        Returns:
            The environment variable value

        Raises:
            ConfigurationError: If the variable is not set
        """
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(
                f"Required environment variable {key} is not set. "
                f"Please check your .env file or environment."
            )
        return value

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")

        if self.max_retries < 0:
            raise ConfigurationError(f"Invalid max_retries: {self.max_retries}")

        if self.retry_delay < 0:
            raise ConfigurationError(f"Invalid retry_delay: {self.retry_delay}")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The configuration object

    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config

    if _config is None:
        _config = Config()
        _config.validate()

    return _config


def reset_config() -> None:
    """Reset the global configuration instance.

    Useful for testing or reloading configuration.
    """
    global _config
    _config = None
