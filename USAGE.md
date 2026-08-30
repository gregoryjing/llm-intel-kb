# llm-intel-kb 完整使用说明

> AI/LLM 领域情报知识库 —— 从启动到日常使用的完整指南。

## 项目亮点

**llm-intel-kb** 是一个面向 AI/LLM 领域的**可验证检索增强生成（RAG）知识库**。它不只是一个能问答的工具，更是一个强调"数据更新后质量不退化"的工程化 RAG 系统。

**为什么这个项目值得一看：**

- **可验证的质量保证** —— 内置金标问答集 + 自动评估 + 回归门禁，数据更新后自动对比基线，退化即告警。这是大多数 RAG 项目忽略的工程环节。
- **多模型即插即用** —— 通过一个 toml 文件切换本地/远程模型（Ollama、dots3、DeepSeek），换模型不改一行代码。
- **版本化数据管理** —— 语料支持快照、回滚，类似 Git 的数据版本控制，确保可追溯。
- **工程完整度** —— CLI + Web UI 双交付、mypy strict 类型检查、102 个单元测试全绿、ruff 零错误。
- **检索质量优化** —— 针对中文混合检索场景做了质量感知排序，解决标题级短文档遮蔽核心内容的问题（MRR 提升 3.4%）。

**技术栈：** Python · FastAPI · BM25 + bge-m3 向量 · RRF 融合 · Typer CLI · 自包含单页前端

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境要求](#2-环境要求)
3. [首次启动](#3-首次启动)
4. [日常使用](#4-日常使用)
5. [Web UI 使用指南](#5-web-ui-使用指南)
6. [CLI 命令详解](#6-cli-命令详解)
7. [模型配置](#7-模型配置)
8. [当前状态与修复记录](#8-当前状态与修复记录)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 项目简介

llm-intel-kb 是一个 AI/LLM 领域的**情报知识库**，自动抓取学术论文（arXiv）、框架 Release（GitHub）、AI 资讯（博客/新闻），
构建可检索的知识库，并通过评估门禁保证数据更新后检索质量不会"悄悄变坏"。

**核心能力：**
- 自动抓取 12 个数据源（arXiv 论文、GitHub Releases、AI 资讯）
- 双语混合检索（BM25 词法 + bge-m3 向量，RRF 融合）
- 内容质量感知排序（自动降权标题级短文档，提升核心内容排位）
- 引用生成（每个事实标注来源 `[n]`）
- 金标集自动评估 + 回归门禁
- 多模型即插即用（本地 Ollama / 远程 API 统一配置）
- CLI + Web UI 双交付

**当前数据规模：** 103 篇文档 · 727 个检索块 · 12 条金标问题

**适用场景：**
- 个人 AI 资讯聚合与问答
- RAG 工程实验（换模型、换检索策略、换 embedding）
- 作品集展示（工程完整度 + 可验证质量）

---

## 2. 环境要求

### 必需
- Python 3.11 或更高（推荐 3.12）
- 已安装本项目的虚拟环境（`.venv`，项目已内置）

### 可选（根据使用场景）
| 组件 | 用途 | 是否必须 |
|------|------|----------|
| Ollama + bge-m3 | 向量检索 | 推荐（已有预计算索引，但新查询需要） |
| Ollama + qwen3-vl:8b | 本地生成对话 | 可选（或用远程 API 替代） |
| dots3 API Key | 远程生成对话 | 可选（填入 `.env` 的 `LLMKB_DOTS3_API_KEY`） |
| PostgreSQL + pgvector | 规模化索引后端 | 可选（默认用本地文件索引） |

---

## 3. 首次启动

### 步骤 1：确认虚拟环境

项目已内置 `.venv`，无需重新创建。在项目目录下打开 PowerShell：

```powershell
Set-Location "C:\Users\Gregoryjing\Desktop\脚本整合\个人项目\llm-intel-kb"
.venv\Scripts\Activate.ps1
```

### 步骤 2：配置 .env

```powershell
Copy-Item .env.example .env
```

用记事本编辑 `.env`，填入你的 API Key（如果有）：

```
LLMKB_DOTS3_API_KEY=你的dots3密钥      # 远程生成模型（推荐）
LLMKB_DEEPSEEK_API_KEY=你的deepseek密钥  # 可选
```

> **注意：** 如果只使用本地 Ollama 模型，可以不填任何 Key。

### 步骤 3：运行体检

```powershell
llm-intel-kb doctor
```

检查各组件连通性：
- ✅ READY = 可用
- ⚠ SKIP = 缺少配置（非必需组件可忽略）
- ❌ FAIL = 需要修复

### 步骤 4：启动 Web UI

```powershell
llm-intel-kb serve --port 8000
```

浏览器打开 **http://127.0.0.1:8000**

### 一键启动（推荐）

已在桌面创建快捷方式 **"llm-intel-kb 知识库"**，双击即可启动 Web UI。详见第 5 节。

---

## 4. 日常使用

### 4.1 数据更新流程

```powershell
# 1. 抓取最新数据（增量抓取，不重复入库）
llm-intel-kb data fetch

# 2. 查看暂存区状态
llm-intel-kb data status

# 3. 提交入库（清洗 + 去重 + 生成新快照）
llm-intel-kb data commit

# 4. LLM 富化（生成中文摘要 + 英文主题词，缓存命中不重复付费）
llm-intel-kb data enrich -l 20    # 只富化前 20 篇，控制成本

# 5. 重建索引（切块 + 向量化）
llm-intel-kb index build

# 6. 评估验证（确保质量没退化）
llm-intel-kb eval run

# 7. 与基线对比（退化则退出码 1）
llm-intel-kb eval gate
```

### 4.2 问答方式

**方式一：Web UI**（推荐）

打开浏览器 http://127.0.0.1:8000，在"问答"标签输入问题。

**方式二：CLI 纯检索**（免费，不耗生成 token）

```powershell
llm-intel-kb index search "vllm 性能优化"
```

**方式三：CLI 检索 + 生成**

```powershell
llm-intel-kb index ask "vLLM 最近有哪些版本更新？"
llm-intel-kb index ask "RAG 最新研究方向" -p dots3    # 指定生成模型
```

**方式四：多模型对比**

```powershell
llm-intel-kb compare "用一句话解释什么是 RAG"
```

同一问题并发发给所有就绪模型，并排展示回答与耗时。

---

## 5. Web UI 使用指南

### 启动方法

**方法一：桌面快捷方式**

双击桌面上的 **"llm-intel-kb 知识库"** 快捷方式。

**方法二：命令行**

```powershell
.venv\Scripts\Activate.ps1
llm-intel-kb serve --port 8000
```

**方法三：项目内脚本**

直接双击项目目录下的 `start-server.bat`。

### 四个功能标签

| 标签 | 功能 |
|------|------|
| **问答** | 输入问题，获得带引用标注的答案；可切换生成模型 |
| **语料** | 浏览所有文档，支持搜索、分页、查看富化摘要 |
| **版本** | 查看快照历史，支持一键回滚到任意版本 |
| **评估** | 查看检索指标（hit@5 / MRR / judge）、指标趋势图、badcase |

### 问答页使用

1. 在输入框输入问题（如 "vLLM 最新版本有什么更新？"）
2. 可选择生成模型（默认使用第一个就绪模型）
3. 点击发送，等待返回带 `[n]` 引用标注的答案
4. 每条答案下方展示检索到的来源块

---

## 6. CLI 命令详解

### 完整命令树

```
llm-intel-kb
├── version          # 显示版本
├── doctor           # 环境体检
├── init-db          # 初始化 pgvector 数据库
├── compare "问题"   # 多模型对比
├── serve --port N   # 启动 Web UI
├── data
│   ├── sources      # 查看数据源注册表
│   ├── fetch        # 抓取数据到暂存区
│   ├── status       # 暂存区状态
│   ├── commit       # 提交入库（生成快照）
│   ├── enrich       # LLM 富化
│   ├── list         # 列出语料
│   ├── inspect      # 查看单篇文档详情
│   ├── log          # 快照历史
│   └── rollback     # 回滚到指定快照
├── index
│   ├── build        # 切块 + 建索引
│   ├── status       # 索引状态
│   ├── search "词"  # 纯检索
│   ├── ask "问题"   # 检索 + 生成
│   └── pg-ann       # pgvector 后端索引
└── eval
    ├── run          # 运行评估
    ├── gate         # 回归门禁
    ├── baseline     # 固化基线
    └── show         # 查看基线
```

### 常用命令参数

```powershell
# 数据抓取：只抓指定源
llm-intel-kb data fetch --source arxiv-cs-ai

# 富化：指定供应商 + 限额
llm-intel-kb data enrich -p local-ollama -l 50

# 索引：指定后端
llm-intel-kb index build --backend local    # 本地文件（默认）
llm-intel-kb index build --backend pg       # pgvector

# 检索：指定返回数量
llm-intel-kb index search "agent" -k 10

# 问答：指定模型 + token 限制
llm-intel-kb index ask "RAG 最新进展" -p dots3 --max-tokens 2048

# 评估：指定生成模型
llm-intel-kb eval run -p dots3
```

---

## 7. 模型配置

### 配置文件：llm_providers.toml

本地模型和远程 API 统一在此配置，**换模型只改 toml，不改代码**。

### 当前已配置供应商

| 名称 | 类型 | 模型 | 部署 | 状态 |
|------|------|------|------|------|
| `dots3` | chat | dots3-note-prev | 远程 API | 需 Key |
| `deepseek` | chat | deepseek-chat | 远程 API | 需 Key |
| `local-ollama` | chat | qwen3-vl:8b | 本地 Ollama | 需 Ollama |
| `local-ollama-embedding` | embedding | bge-m3 | 本地 Ollama | 需 Ollama |

### 添加新模型

在 `llm_providers.toml` 添加：

```toml
[new-model]
kind = "chat"                   # chat / embedding
base_url = "https://api.example.com"
api_key = "env:LLMKB_NEW_KEY"   # 从环境变量读
model = "model-name"
```

### 本地 Ollama 安装

```powershell
# 安装 Ollama 后执行：
ollama pull qwen3-vl:8b     # 本地生成模型（~5.5GB）
ollama pull bge-m3          # 向量模型（~1.2GB）
```

---

## 8. 当前状态与修复记录

### 检索质量修复（2026-08-30）

**问题：** qbitai（量子位）标题级源平均只有 20 字符 search_text，BM25 对短文档的长度归一化偏爱让这些"标题党"文档挤占 top-1，遮蔽了真正有内容的文档。

**修复：** 在 `HybridSearcher` 中引入内容质量感知的 RRF 分数惩罚：
- search_text < 100 字符的文档，RRF 分数乘以 0.5~0.9 的系数
- 惩罚只降权不过滤，用户明确查"行业动态"时仍可召回

**效果：**

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| MRR | 0.8125 | **0.8403** |
| hit@5 | 100% | 100% |
| 单元测试 | 88 | **102**（+14） |

**具体改善：**
- g001（vLLM 更新）：rank 2 → 1，不再被 qbitai 标题遮蔽
- g005/g006/g007/g010：qbitai 从 top-3 中减少

### 当前评估基线

| 指标 | 数值 |
|------|------|
| 检索命中率 hit@5 | 100%（12/12） |
| MRR | 0.8403 |
| 关键词覆盖率 | 91.7% |
| 质量门禁 | ruff 0 错误 · mypy strict 通过 · 102 测试全绿 |

---

## 9. 常见问题排查

### Q: 启动后问答返回"检索无命中"

**A:** 可能原因：
1. 索引未构建 → 运行 `llm-intel-kb index build`
2. 查询用词太生僻 → 换个说法试试
3. 数据未抓取 → 运行 `llm-intel-kb data fetch && llm-intel-kb data commit`

### Q: doctor 显示 embedding 供应商 FAIL

**A:** 需要本地 Ollama 运行 bge-m3：
```powershell
ollama pull bge-m3
ollama serve
```
或者配置远程 embedding 供应商。

### Q: 生成答案很慢（>30s）

**A:** 本地模型首次加载慢，后续会快。或者切换到远程 API：
```powershell
llm-intel-kb index ask "问题" -p dots3
```

### Q: 如何更新数据

```powershell
llm-intel-kb data fetch       # 抓取新数据
llm-intel-kb data commit      # 提交入库
llm-intel-kb index build      # 重建索引
llm-intel-kb eval gate        # 确认质量没退化
```

### Q: 如何回滚到旧版本

```powershell
llm-intel-kb data log                  # 查看快照历史
llm-intel-kb data rollback <快照ID>    # 回滚
```

### Q: Web UI 打开后无法访问

**A:** 检查：
1. 服务是否启动（PowerShell 中看到 "Uvicorn running on..."）
2. 端口是否被占用 → 换端口 `llm-intel-kb serve --port 8080`
3. 防火墙是否放行

---

## 附录：项目结构

```
llm-intel-kb/
├── src/llm_intel_kb/
│   ├── pipeline/      # 数据面：抓取、清洗、富化、版本化
│   ├── index/         # 服务面：切块、BM25、向量、RRF 检索
│   ├── rag/           # 引用生成
│   ├── eval/          # 评估：金标集、指标、门禁
│   ├── llm/           # 多模型供应商抽象
│   ├── web/           # FastAPI 后端 + 单页前端
│   └── cli/           # 命令行入口
├── data/              # 运行时数据（语料、快照、索引、评估）
├── golden/qa.toml     # 金标问答集
├── data_sources.toml  # 数据源注册表
├── llm_providers.toml # 模型供应商配置
└── tests/             # 单元测试 + 集成测试
```
