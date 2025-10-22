from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangExtractConfig(BaseSettings):
    """Configuration for LangExtract entity extraction service."""

    model_config = SettingsConfigDict(
        env_prefix="LANGEXTRACT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str
    model_id: str = Field(
        default="gemini-2.5-flash", description="Model identifier for extraction"
    )
    extraction_passes: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of extraction passes for improved recall",
    )
    max_workers: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of parallel workers for processing",
    )
    max_char_buffer: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum character buffer size for context windows",
    )
