from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_intel_kb import __version__
from llm_intel_kb.checks import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    load_provider_list,
    run_checks,
)
from llm_intel_kb.cli.data import data_app
from llm_intel_kb.cli.eval_cli import eval_app
from llm_intel_kb.cli.guard_cli import guard_app
from llm_intel_kb.cli.index_cli import index_app
from llm_intel_kb.config import get_settings
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.store.db import create_db_engine, ensure_pgvector

app = typer.Typer(
    help="AI/LLM 领域情报知识库：数据管线与运维 CLI",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(guard_app, name="guard")
console = Console()

_STATUS_LABELS = {
    STATUS_OK: "[green]OK[/green]",
    STATUS_FAIL: "[red]FAIL[/red]",
    STATUS_SKIP: "[yellow]SKIP[/yellow]",
}


@app.command()
def version() -> None:
    """输出版本号."""
    console.print(f"llm-intel-kb {__version__}")


@app.command()
def doctor() -> None:
    """环境体检：数据库 / pgvector / 所有已配置模型供应商."""
    results = run_checks(get_settings())
    table = Table(title="llm-intel-kb doctor")
    table.add_column("检查项", style="bold")
    table.add_column("状态", justify="center")
    table.add_column("详情", overflow="fold")
    for result in results:
        label = _STATUS_LABELS.get(result.status, result.status)
        table.add_row(result.name, label, result.detail)
    console.print(table)
    if any(result.status == STATUS_FAIL for result in results):
        console.print("[red]存在 FAIL 项，退出码 1[/red]")
        raise typer.Exit(code=1)


@app.command("init-db")
def init_db() -> None:
    """初始化数据库：安装 pgvector 扩展（幂等）."""
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    try:
        pgvector_version = ensure_pgvector(engine)
    finally:
        engine.dispose()
    console.print(f"[green]OK[/green] pgvector {pgvector_version} 就绪")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="开发模式：代码变更自动重启"),
) -> None:
    """启动 Web 服务：浏览器打开 http://<host>:<port> 即可使用."""
    import uvicorn

    console.print(f"[green]启动[/green] http://{host}:{port}（Ctrl+C 停止）")
    uvicorn.run(
        "llm_intel_kb.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def compare(
    question: str = typer.Argument(..., help="要发给所有对话模型的同一个问题"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="每个模型的最大回复长度"),
    timeout: float = typer.Option(180.0, "--timeout", help="单个请求的超时秒数"),
) -> None:
    """模型对比：同一个问题并发发给所有已配置的对话模型，并排展示回答."""
    settings = get_settings()
    all_providers = load_provider_list(settings)
    chat_providers = [p for p in all_providers if p.supports_chat and p.is_ready]
    skipped = [p for p in all_providers if p.supports_chat and not p.is_ready]

    if not chat_providers:
        console.print("[red]没有就绪的对话模型[/red]：请在 llm_providers.toml 中配置")
        raise typer.Exit(code=1)

    for provider in skipped:
        console.print(f"[yellow]跳过 {provider.name}[/yellow]：缺少 API Key")

    console.print(f"[bold]问题[/bold]：{question}\n")

    def ask(provider: ProviderConfig) -> tuple[ProviderConfig, str | None, Exception | None, float]:
        start = time.perf_counter()
        try:
            reply = llm_client.chat(
                provider,
                [{"role": "user", "content": question}],
                timeout,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return provider, None, exc, time.perf_counter() - start
        return provider, reply, None, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=len(chat_providers)) as executor:
        results = list(executor.map(ask, chat_providers))

    panels = []
    had_failure = False
    for provider, reply, error, elapsed in results:
        if error is not None:
            had_failure = True
            body = f"[red]请求失败[/red]\n{type(error).__name__}: {error}"
        else:
            body = reply or "[dim]（空回复）[/dim]"
        location = "本地" if provider.is_local else "远程"
        panels.append(
            Panel(
                body,
                title=f"[bold]{provider.name}[/bold] · {provider.model}",
                subtitle=f"{location} · {elapsed:.1f}s",
            )
        )
    console.print(Columns(panels, equal=True, expand=True))

    if had_failure:
        console.print("[red]存在失败请求，退出码 1[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
