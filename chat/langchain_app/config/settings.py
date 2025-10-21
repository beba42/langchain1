"""Project-wide configuration settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Settings for local agents and tools."""

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.1:8b"
    search_enabled: bool = True

    class Config:
        env_prefix = "LC_"


settings = AppSettings()
