"""
Application configuration using pydantic-settings.

WHY PYDANTIC-SETTINGS?
- Type-safe configuration from environment variables
- Automatic validation and type coercion (e.g., "true" → True)
- Follows the 12-factor app methodology: config lives in the environment, not code
- Single source of truth: one Settings object used everywhere

HOW IT WORKS UNDER THE HOOD:
- BaseSettings reads from env vars, .env files, and constructor args
- Priority: constructor args > env vars > .env file > defaults
- Pydantic v2 uses pydantic-core (written in Rust) for 5-50x faster validation
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    For example, set APP_NAME=MyApp in your environment or .env file.
    """

    # Application
    app_name: str = "Knowledge Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # File Upload
    upload_dir: Path = Path("./uploads")
    max_file_size_mb: int = 10
    allowed_extensions: set[str] = {".txt", ".md", ".pdf"}

    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    model_config = {
        # Look for a .env file in the project root
        "env_file": ".env",
        # Allow extra fields from env vars without raising validation errors
        "extra": "ignore",
        # Case-insensitive env var matching
        "case_sensitive": False,
    }


# Singleton pattern — create one instance, import everywhere
# This is evaluated once when the module is first imported
settings = Settings()
