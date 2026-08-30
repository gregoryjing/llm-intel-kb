"""CLI 共享辅助：供应商加载与数据源路径解析."""

from __future__ import annotations

from pathlib import Path

import typer

from llm_intel_kb.config import get_settings
from llm_intel_kb.llm.providers import ProviderConfig, load_providers

DEFAULT_SOURCES_PATH = Path("data_sources.toml")


def load_provider_list_or_exit(need: str = "chat") -> list[ProviderConfig]:
    """加载全部供应商；一个都没有时给出指引并退出."""
    settings = get_settings()
    toml_path = settings.providers_file or Path("llm_providers.toml")
    providers = load_providers(toml_path, settings)
    if not providers:
        raise typer.BadParameter(
            "未配置任何模型供应商：请在 llm_providers.toml 声明（本地 Ollama 无需 Key）"
        )
    return providers


def resolve_sources_path() -> Path:
    settings = get_settings()
    return settings.sources_file or DEFAULT_SOURCES_PATH
