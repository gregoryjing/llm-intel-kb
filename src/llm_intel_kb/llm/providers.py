from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

from llm_intel_kb.config import Settings

KIND_CHAT = "chat"
KIND_EMBEDDING = "embedding"
KIND_BOTH = "both"
_VALID_KINDS = {KIND_CHAT, KIND_EMBEDDING, KIND_BOTH}

ADAPTER_OPENAI = "openai"
ADAPTER_OLLAMA = "ollama"
ADAPTER_ANTHROPIC = "anthropic"
_VALID_ADAPTERS = {ADAPTER_OPENAI, ADAPTER_OLLAMA, ADAPTER_ANTHROPIC}

DEFAULT_PROVIDERS_FILE = Path("llm_providers.toml")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


@dataclass(frozen=True)
class ProviderConfig:
    """一个模型供应商（远程 API 或本地服务）."""

    name: str
    kind: str  # "chat" | "embedding" | "both"
    base_url: str
    model: str
    api_key: str | None = None  # 支持 "env:VAR" 语法引用环境变量
    adapter: str = ADAPTER_OPENAI  # "openai" | "ollama"（原生接口）| "anthropic"（Messages API）
    num_ctx: int | None = None  # ollama 原生接口的上下文窗口（思考型模型需要更大窗口）
    skip_think: bool = False  # 跳过思考直接输出正文：ollama=预置空 <think> 块；anthropic=thinking disabled

    @property
    def supports_chat(self) -> bool:
        return self.kind in (KIND_CHAT, KIND_BOTH)

    @property
    def supports_embedding(self) -> bool:
        return self.kind in (KIND_EMBEDDING, KIND_BOTH)

    @property
    def is_local(self) -> bool:
        host = (urlparse(self.base_url).hostname or "").lower()
        return host in _LOCAL_HOSTS

    @property
    def is_ollama_native(self) -> bool:
        return self.adapter == ADAPTER_OLLAMA

    @property
    def is_anthropic(self) -> bool:
        return self.adapter == ADAPTER_ANTHROPIC

    def messages_url(self) -> str:
        """anthropic Messages API 地址：兼容根域名 / 带 /v1 / 完整路径三种写法."""
        url = self.base_url.rstrip("/")
        if url.endswith("/v1/messages"):
            return url
        if url.endswith("/v1"):
            return f"{url}/messages"
        return f"{url}/v1/messages"

    def native_base_url(self) -> str:
        """ollama 原生接口地址：OpenAI 兼容地址去掉结尾的 /v1."""
        if self.base_url.rstrip("/").endswith("/v1"):
            return self.base_url.rstrip("/")[: -len("/v1")]
        return self.base_url.rstrip("/")

    def resolved_api_key(self) -> str | None:
        """解析 "env:VAR" 语法；返回 None 表示无可用 Key."""
        if self.api_key is None:
            return None
        if self.api_key.startswith("env:"):
            return os.environ.get(self.api_key[4:]) or None
        return self.api_key

    @property
    def is_ready(self) -> bool:
        """本地服务无需 Key 即就绪；远程 API 必须有 Key."""
        return self.is_local or bool(self.resolved_api_key())


def load_providers(toml_path: Path | None, settings: Settings) -> list[ProviderConfig]:
    """优先读 providers.toml；无该文件时回退到 .env 单供应商配置."""
    _bridge_env_file()
    if toml_path is not None and toml_path.exists():
        return _from_toml(toml_path)
    return _from_settings(settings)


def _bridge_env_file() -> None:
    """把 .env 的键值注入 os.environ（真实环境变量优先）.

    pydantic-settings 只把 .env 填充进 Settings 字段，不写进程环境；
    而供应商的 "env:VAR" 语法读的是 os.environ——不桥接的话
    .env 里的 API Key 对多供应商模式永远不可见（真实踩坑）。
    """
    for key, value in dotenv_values(".env").items():
        if value is not None and key not in os.environ:
            os.environ[key] = value


def _from_toml(path: Path) -> list[ProviderConfig]:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    providers: list[ProviderConfig] = []
    for name, section in raw.items():
        if not isinstance(section, dict):
            continue
        kind = str(section.get("kind", KIND_CHAT))
        if kind not in _VALID_KINDS:
            raise ValueError(f"供应商 {name} 的 kind 非法：{kind}（应为 chat/embedding/both）")
        adapter = str(section.get("adapter", ADAPTER_OPENAI))
        if adapter not in _VALID_ADAPTERS:
            raise ValueError(
                f"供应商 {name} 的 adapter 非法：{adapter}（应为 openai/ollama/anthropic）"
            )
        base_url = section.get("base_url")
        model = section.get("model")
        if not isinstance(base_url, str) or not isinstance(model, str):
            raise ValueError(f"供应商 {name} 缺少 base_url 或 model")
        api_key = section.get("api_key")
        num_ctx = section.get("num_ctx")
        skip_think = section.get("skip_think", False)
        providers.append(
            ProviderConfig(
                name=name,
                kind=kind,
                base_url=base_url,
                model=model,
                api_key=api_key if isinstance(api_key, str) else None,
                adapter=adapter,
                num_ctx=int(num_ctx) if isinstance(num_ctx, int) else None,
                skip_think=bool(skip_think),
            )
        )
    return providers


def _from_settings(settings: Settings) -> list[ProviderConfig]:
    providers: list[ProviderConfig] = []
    if settings.llm_configured:
        assert settings.llm_base_url and settings.llm_model and settings.llm_api_key
        providers.append(
            ProviderConfig(
                name="default",
                kind=KIND_CHAT,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
            )
        )
    if settings.embedding_configured:
        assert (
            settings.embedding_base_url and settings.embedding_model and settings.embedding_api_key
        )
        providers.append(
            ProviderConfig(
                name="default-embedding",
                kind=KIND_EMBEDDING,
                base_url=settings.embedding_base_url,
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,
            )
        )
    return providers
