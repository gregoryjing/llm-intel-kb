from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，优先级：环境变量 > .env > 默认值."""

    model_config = SettingsConfigDict(
        env_prefix="LLMKB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/llm_intel_kb",
        description="PostgreSQL（pgvector）连接串",
    )

    llm_base_url: str | None = Field(default=None, description="OpenAI 兼容 Chat API 地址")
    llm_api_key: str | None = Field(default=None, description="Chat API 密钥")
    llm_model: str | None = Field(default=None, description="Chat 模型名")
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0)

    embedding_base_url: str | None = Field(default=None, description="Embedding API 地址")
    embedding_api_key: str | None = Field(default=None, description="Embedding API 密钥")
    embedding_model: str | None = Field(default=None, description="Embedding 模型名")

    data_dir: Path = Field(default=Path("data"), description="语料与快照的本地存储目录")

    providers_file: Path | None = Field(
        default=None, description="多供应商配置文件路径（默认找当前目录 llm_providers.toml）"
    )

    sources_file: Path | None = Field(
        default=None, description="数据源注册表路径（默认找当前目录 data_sources.toml）"
    )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_base_url and self.embedding_api_key and self.embedding_model)


def get_settings() -> Settings:
    """每次调用读取新配置，便于测试与长跑进程中的配置刷新."""
    return Settings()
