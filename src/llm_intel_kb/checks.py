from __future__ import annotations

from dataclasses import dataclass

from llm_intel_kb.config import Settings
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import DEFAULT_PROVIDERS_FILE, ProviderConfig, load_providers
from llm_intel_kb.store.db import create_db_engine, probe_database

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def load_provider_list(settings: Settings) -> list[ProviderConfig]:
    toml_path = settings.providers_file or DEFAULT_PROVIDERS_FILE
    return load_providers(toml_path, settings)


def check_database(settings: Settings) -> list[CheckResult]:
    engine = create_db_engine(settings.database_url)
    try:
        info = probe_database(engine)
    except Exception as exc:
        return [CheckResult("数据库连接", STATUS_FAIL, _format_error(exc))]
    finally:
        engine.dispose()
    results = [CheckResult("数据库连接", STATUS_OK, f"PostgreSQL {info['server_version']}")]
    if info["pgvector_version"]:
        results.append(
            CheckResult("pgvector 扩展", STATUS_OK, f"vector {info['pgvector_version']}")
        )
    else:
        results.append(
            CheckResult(
                "pgvector 扩展",
                STATUS_FAIL,
                "未安装：运行 llm-intel-kb init-db 或改用 pgvector 官方镜像",
            )
        )
    return results


def check_chat_provider(provider: ProviderConfig) -> CheckResult:
    name = f"对话模型 · {provider.name}"
    if not provider.is_ready:
        return CheckResult(
            name, STATUS_SKIP, f"{provider.model} 缺少 API Key（{provider.base_url}）"
        )
    try:
        probe = llm_client.ping_chat(provider)
    except Exception as exc:
        return CheckResult(name, STATUS_FAIL, _format_error(exc))
    location = "本地" if provider.is_local else "远程"
    return CheckResult(
        name,
        STATUS_OK,
        f"{probe.model} · {location} · {probe.latency_ms}ms · 回复「{probe.reply}」",
    )


def check_embedding_provider(provider: ProviderConfig) -> CheckResult:
    name = f"向量模型 · {provider.name}"
    if not provider.is_ready:
        return CheckResult(
            name, STATUS_SKIP, f"{provider.model} 缺少 API Key（{provider.base_url}）"
        )
    try:
        probe = llm_client.ping_embedding(provider)
    except Exception as exc:
        return CheckResult(name, STATUS_FAIL, _format_error(exc))
    location = "本地" if provider.is_local else "远程"
    return CheckResult(
        name,
        STATUS_OK,
        f"{probe.model} · {location} · 维度 {probe.dimension} · {probe.latency_ms}ms",
    )


def check_providers(settings: Settings) -> list[CheckResult]:
    results: list[CheckResult] = []
    for provider in load_provider_list(settings):
        if provider.supports_chat:
            results.append(check_chat_provider(provider))
        if provider.supports_embedding:
            results.append(check_embedding_provider(provider))
    if not results:
        results.append(
            CheckResult("模型供应商", STATUS_SKIP, "未配置任何模型（llm_providers.toml 或 .env）")
        )
    return results


def run_checks(settings: Settings) -> list[CheckResult]:
    return [*check_database(settings), *check_providers(settings)]


def _format_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message if len(message) <= 160 else message[:157] + "..."
