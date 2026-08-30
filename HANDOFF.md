# llm-intel-kb 项目全流程交接文档

> **本文档用途**：完整交接本项目的目的、架构、技术选型、全流程与所有关键工程决策。
> 面向对象：接手后续开发的 AI 模型或工程师。读完本文应能独立理解、运行、修改本项目，
> 且不重复踩已踩过的坑。配套的 `README.md` 面向人类读者（快速上手 + 作品集展示），
> 本文档面向接手者（完整原理 + 决策依据），两者互补。

---

## 1. 项目定位与设计目的

**一句话定位**：AI/LLM 领域情报知识库 —— 数据集生命周期管线 × 工程化 RAG × 评估门禁。

**它回答的真实问题**：当你的知识库数据每天都在变（新论文、新版本、新资讯），你怎么保证
更新之后系统没有悄悄变坏？

这不是一个"又一个 RAG demo"。市面上的 RAG 教程项目几乎都停在"能问答"，
本项目把重心放在**数据可靠性**上，与检索效果同等地位。核心主张是：

> 数据变更（重新抓取、重新切块、换模型、换检索策略）不能靠肉眼抽查验收，
> 必须有一条**可复现的自动评估 + 回归门禁**链路：金标问答集 → 自动评估 →
> 与基线对比 → 指标退化即拦截（退出码 1）→ badcase 人工复盘后回流金标集。

**性质**：作品集项目（求职展示），面向 AI 应用工程师 / AI 产品方向岗位。
因此设计上刻意突出工程完整度：版本化数据、多模型可切换、评估门禁、CLI + Web 双交付、
mypy strict + 102 个单测全绿。所有结论都有 data/evals/ 下的报告可复现。

---

## 2. 系统架构总览

四个里程碑对应四个"面"，数据单向流动：

```
┌───────────────────── 数据面（M1）─────────────────────┐
  数据源注册表          暂存区          内容哈希去重        快照（版本化）
  RSS / GH Releases ─fetch─▶ staging ─commit─▶ corpus(不可变) ─▶ snapshot_N
  （data_sources.toml）  （可丢弃）      清洗+语种检测            ↑ parent 指针
                                                         snapshot_N-1
  └──────── LLM 富化（中文摘要+英文主题词，缓存命中不重复付费）──┘

┌───────────────────── 服务面（M2）─────────────────────┐
  snapshot ─chunk─▶ 结构感知切块（段落边界+标题前缀+相邻重叠）
                   ├──▶ BM25 词法索引（纯 Python，中英混合分词）
                   └──▶ 向量索引（本地文件 / pgvector HNSW 二选一）
                             └─search─▶ RRF 融合排序 ─ask─▶ 引用生成 [n]

┌───────────────────── 评估面（M3）─────────────────────┐
  golden/qa.toml ─eval run─▶ hit@5 / MRR / 关键词覆盖 / LLM-as-judge
                  └─eval gate─▶ 与基线对比，指标退化即拒绝（退出码 1）
                  └─badcase 回流─▶ 人工复盘后追加进金标集

┌───────────────────── 交付（M4）───────────────────────┐
  CLI（typer + rich）× Web UI（FastAPI + 自包含单页，无 CDN、断网可用）
```

里程碑与状态：M0 骨架/配置/doctor/多供应商/compare ✅ · M1 数据面 ✅ ·
M2 混合检索 + pgvector 可选后端 ✅ · M3 评估门禁 ✅ · M4 Web UI ✅。全部完成。

---

## 3. 技术栈清单

| 层 | 技术 | 用途 |
|---|---|---|
| 语言/构建 | Python 3.11+ / hatchling | `uv pip install -e ".[dev]"` |
| CLI | typer + rich | 命令行界面与表格渲染（`llm-intel-kb` 入口） |
| Web | FastAPI + uvicorn | HTTP API + 单页前端 |
| HTTP | httpx | Anthropic 分支的直接调用 |
| LLM SDK | openai（官方 SDK） | OpenAI 兼容接口分支 |
| 抓取 | feedparser | RSS / Atom 解析（GitHub Releases 走 REST API） |
| 数据库 | SQLAlchemy 2 + psycopg3 + pgvector | 可选 pgvector 后端（HNSW ANN） |
| 配置 | pydantic-settings | `LLMKB_` 前缀环境变量 + .env |
| 检索 | 自研纯 Python BM25 + 余弦相似度 + RRF | 无重依赖、行为完全可控可测 |
| 模型运行时 | Ollama（本地）+ 远程 API | qwen3-vl:8b（chat）/ bge-m3（embedding）/ dots3-note-prev（远程 chat） |
| 质量 | ruff（line-length=120）+ mypy strict + pytest | 门禁全绿 |

设计取舍说明：检索层**刻意不引入** elasticsearch / faiss / chromadb 等重依赖——
语料量级（百篇文档、千级 chunk）下纯 Python 实现（BM25 + 暴力余弦）毫秒级返回，
换来的是：零外部服务依赖、行为可单测、指标退化可归因。
pgvector 作为可选后端保留（`index pg-ann`），演示"数据量上来后如何平滑换引擎"。

---

## 4. 代码结构导览

```
llm-intel-kb/
├── llm_providers.toml          # ★ 模型供应商注册表（chat/embedding × 三种 adapter）
├── data_sources.toml           # ★ 数据源注册表（含 min_chars 分级注释，badcase 结论沉淀处）
├── golden/qa.toml              # ★ 金标问答集（12 条，含 badcase 回流的守护问题）
├── docker-compose.yml          # pgvector 数据库一键启动
├── Makefile / pyproject.toml   # lint / typecheck / test 入口
├── src/llm_intel_kb/
│   ├── config.py               # pydantic-settings；LLMKB_ 前缀；优先级 环境变量 > .env > 默认
│   ├── checks.py               # doctor 体检（数据库/pgvector/各供应商连通性）
│   ├── cli/
│   │   ├── main.py             # version / doctor / init-db / serve / compare
│   │   ├── data.py             # data sources|fetch|status|commit|enrich|list|inspect|log|rollback
│   │   ├── index_cli.py        # index build|status|pg-ann|search|ask
│   │   └── eval_cli.py         # eval run|gate|baseline|show
│   ├── pipeline/               # 数据面
│   │   ├── sources.py          # SourceSpec 解析（rss / github-releases 两种 type）
│   │   ├── clean.py            # 清洗 + 语种检测 + min_chars 门槛
│   │   ├── enrich.py           # LLM 富化（摘要+主题词，按内容哈希缓存）
│   │   ├── commit.py           # staging → corpus 提交，生成 snapshot
│   │   ├── store.py            # corpus 快照读写 / parent 链 / rollback
│   │   └── models.py           # CorpusDocument / Snapshot 等数据模型
│   ├── index/                  # 服务面
│   │   ├── chunking.py         # 结构感知切块（Chunk 数据模型）
│   │   ├── lexical.py          # BM25（中英混合分词）
│   │   ├── embeddings.py       # EmbeddingBackend 抽象 + cosine_similarity
│   │   ├── search.py           # ★ HybridSearcher：BM25+向量 → RRF → 同文档限流
│   │   ├── store_local.py      # 本地文件索引（chunks.jsonl + vectors.json）
│   │   ├── store_pg.py         # pgvector HNSW 后端（接口与本地文件对齐）
│   │   └── loader.py           # 索引加载（含按 snapshot 校验）
│   ├── rag/answer.py           # 引用生成（[n] 标注）+ extract_citations
│   ├── llm/
│   │   ├── providers.py        # ★ ProviderConfig：三 adapter / env: 语法 / .env 桥接
│   │   └── client.py           # ★ chat() 分派：openai / ollama / anthropic 三分支
│   ├── eval/
│   │   ├── golden.py           # 金标集解析
│   │   ├── runner.py           # ★ run_eval + LLM-as-judge + judge 容错解析 + badcase 导出
│   │   └── gate.py             # ★ 回归门禁（阈值 + 基线对比）
│   ├── web/
│   │   ├── app.py              # FastAPI：11 个 API 端点
│   │   └── static/index.html   # 自包含单页（问答/语料/版本/评估四个 tab，无 CDN）
│   └── store/db.py             # 数据库引擎（connect_timeout=5，见踩坑 9）
├── tests/unit/（9 个文件 84 用例）/ tests/integration/（真实网络，默认排除）
├── scripts/                    # show_eval.py / web_smoke.py
└── data/                       # 运行时数据（见 §7 数据资产）
```

★ 标记的文件是理解本项目的核心，接手后建议按此顺序阅读：
`providers.py → client.py → search.py → runner.py → gate.py`。

---

## 5. 全流程详解（一条数据的完整生命周期）

### 5.1 数据面（M1）：fetch → clean → enrich → commit → snapshot

1. **注册**：`data_sources.toml` 每段一个源。字段：`type`（rss / github-releases）、
   `url`/`repo`、`max_items`（单次抓取上限，增量抓取，抓过的不重复入库）、
   `min_chars`（清洗后正文最短字符数，低于则丢弃）、`enabled`（false = 保留配置但跳过）。
2. **fetch**：`llm-intel-kb data fetch` 抓取 enabled 源 → `data/staging/<source_id>.json`。
   staging 是可丢弃的暂存区，随时可以重抓。
3. **clean**：清洗 + 语种检测；**min_chars 门槛在这里生效**（分级入库，见 §8 踩坑 2）。
4. **enrich**：`data enrich` 调 LLM 为每篇文档生成中文摘要 + 英文主题词，
   **按内容哈希缓存**（`data/enrichment/xx/<hash>.json`），已富化过的文档不重复付费。
5. **commit**：`data commit` 把 staging 提交进 corpus——
   **corpus 不可变**：每篇文档按内容哈希寻址存 `data/corpus/<前2位>/<完整hash>.json`，
   内容相同即同一文件（天然去重）；同时生成快照 `data/snapshots/<时间戳>.json`
   （记录本快照包含哪些文档哈希 + parent 指向上一个快照），`data/current.json` 指向最新。
6. **版本操作**：`data log` 看快照链、`data rollback` 回滚到指定快照（只改 current 指针，
   corpus 本身不动——这就是"不可变"带来的零成本回滚）。

### 5.2 服务面（M2）：chunk → 双索引 → RRF 混合检索 → 引用生成

1. **build**：`index build` 对当前快照做结构感知切块：
   尊重段落边界、每块带标题前缀（保留文档结构信息）、相邻块重叠（避免语义被切断）。
   产出 `data/index/chunks.jsonl`（全部 chunk）+ `vectors.json`（chunk_id → 向量）。
2. **检索**：`HybridSearcher.search(query, k=5)`：
   - 先取宽：`fetch_k = max(k*4, 20)`，BM25 和向量各取前 fetch_k；
   - **RRF 融合**（K=60）：`score(c) = Σ 1/(60 + rank_branch(c))`，两分支都命中的块得分累加；
   - **内容质量感知排序**（2026-08-30 新增）：根据每个 chunk 的 search_text 长度计算质量乘数（0.5~1.0），标题级短文档（<100字符）的 RRF 分数乘以该系数降权，提升核心内容排位。MRR 从 0.812 → 0.840；
   - **同文档限流**：`MAX_PER_DOC = 3`，单一文档最多占 3 个席位，保证 top-k 多样性；
   - 每个 hit 同时记录 `lexical_rank` / `vector_rank`（1-based，None = 该分支未命中），
     方便归因"这个块是靠词法还是语义进来的"。
   - **自动退化**：无向量（未配 embedding 供应商）时自动退化为纯词法，`mode` 属性如实标注。
3. **生成**：`rag/answer.py` 把检索块编号拼进上下文（`[1] 标题\n来源：...\n正文`，
   **上限 5000 字符**——为本地模型 num_ctx=3072 预算控制，见踩坑 4），
   system prompt 要求"只依据资料回答、事实后标 [n]、资料不足直说不知道"。
   `extract_citations` 用正则 `r"\[(\d+)\]"` 提取引用编号（去重保序）。

### 5.3 评估面（M3）：金标集 → 自动评估 → 回归门禁 → badcase 回流

这是本项目区别于普通 RAG demo 的核心，**接手后改动任何检索/生成逻辑，必须跑这条链**。

1. **金标集** `golden/qa.toml`：每条含 `id` / `question`（贴近真实问法）/
   `expected_sources`（检索 top-k 内命中任一即 hit）/
   `expected_keywords`（生成层关键词覆盖用）/ `notes`。
2. **eval run**：`llm-intel-kb eval run [-p <provider>]` 逐条跑检索+生成：
   - 检索指标不需要任何模型（hit@k、MRR、关键词命中）；
   - 不传 `-p` 只算检索指标；传 `-p dots3` 等则追加生成 + LLM-as-judge；
   - 生成失败（超时/异常）只降级该条的生成指标，检索层照常计算（单条故障不拖垮整场评估）；
   - 报告存 `data/evals/eval_<snapshot>_<时间戳>.json`，未通过条目导出
     `data/badcases/*.jsonl`（含 review_hint 提示人工复盘方向）。
3. **eval gate**：`llm-intel-kb eval gate` 与 `data/evals/baseline.json` 对比：
   - 绝对阈值：hit_rate ≥ 0.7，judge 均分 ≥ 3.0；
   - 相对基线：hit_rate / MRR 退化超过 0.001 即拦截；
   - **逐条回归**：基线通过、当前失败的条目数 > 0 即拦截（默认容忍 0 条）；
   - 拦截 = 退出码 1，可接入 CI 阻断。
   - `eval baseline` 把某次报告设为新基线。
4. **badcase 回流**：人工复盘 badcase（是检索没找到，还是回答质量差？）→
   确认是"系统应答对却没答对"的问题 → **追加**进 golden/qa.toml 作为守护问题。
   金标集只增不改（历史评估才能跨版本对比）。当前 12 条中 g011/g012 就是
   2026-08-29 badcase 回流新增的守护问题。

### 5.4 交付面（M4）：CLI + Web UI

CLI 命令全集（入口 `llm-intel-kb`，typer 子命令组）：

```
llm-intel-kb version / doctor / init-db / serve [--port] / compare "问题"
llm-intel-kb data sources | fetch | status | commit | enrich | list | inspect | log | rollback
llm-intel-kb index build | status | pg-ann | search "查询" | ask "问题" [-p provider]
llm-intel-kb eval run [-p provider] | gate | baseline | show
```

- `doctor`：环境体检（数据库/pgvector/全部已配置供应商连通性），有 FAIL 退出码 1。
- `compare`：同一问题**并发**发给所有就绪对话模型（ThreadPoolExecutor），rich 并排展示，
  标注本地/远程与耗时——这是多供应商抽象的最直观演示。
- `serve`：启动 Web UI。

Web API 端点（`web/app.py`）：
`GET /`（单页）、`/api/status`、`POST /api/search`、`POST /api/ask`、`GET /api/providers`、
`GET /api/corpus`、`GET /api/corpus/{doc_id}`、`GET /api/snapshots`、
`POST /api/snapshots/rollback`、`GET /api/eval/latest`、`GET /api/eval/history`、
`POST /api/index/reload`。

前端（`static/index.html`）：自包含单页、无 CDN 引用、断网可用；四个 tab
（问答 / 语料 / 版本 / 评估）。评估页含**指标趋势图**（SVG，历次评估的
hit/MRR/judge 曲线）和**生成模型对比表**（按 judge_model 分组取各模型最新一次评估，
本地/远程徽标、最优指标高亮）。

---

## 6. 多模型供应商抽象（llm_providers.toml + llm/ 模块）

**设计原则（硬约束）**：换模型只改 toml 一行，不改代码。同时支持本地（Ollama）与
远程 API（DeepSeek / dots3 等）做对比——这是项目的展示重点之一。

### 6.1 配置格式

```toml
[dots3]
kind = "chat"                  # chat（对话）/ embedding（向量化）
adapter = "anthropic"          # 缺省 = openai 兼容；ollama / anthropic 见下
base_url = "https://note3-prev-api.askdiandian.com"
api_key = "env:LLMKB_DOTS3_API_KEY"   # env: 前缀 = 从环境变量读，密钥不进 git
model = "dots3-note-prev"
skip_think = true              # 思考型模型的思考开关（见 6.3）
```

当前注册：`deepseek`（chat/openai 兼容）、`dots3`（chat/anthropic）、
`local-ollama`（chat/ollama，qwen3-vl:8b）、`local-ollama-embedding`（embedding，bge-m3）、
`siliconflow-embedding`（注释掉的远程 embedding 示例）。

### 6.2 三种 adapter 的差异（`client.py` 的 `chat()` 分派）

| adapter | 协议 | 关键差异 |
|---|---|---|
| （缺省）openai | OpenAI 兼容 `/chat/completions` | openai SDK 直连；远程 API 或本地 OpenAI 兼容端点通用 |
| ollama | 原生 `/api/chat` | 对思考型模型更稳；本地模型超时下限放宽到 600s（推理慢是常态不是异常） |
| anthropic | `/v1/messages` | **system 必须放请求体顶层 `system` 字段**（不能混在 messages）；`x-api-key` + `anthropic-version: 2023-06-01` 头认证；base_url 兼容根域名 / 带 /v1 / 完整路径三种写法 |

### 6.3 思考型模型的坑（skip_think 的由来）

qwen3 / dots3 都是思考型模型，**不关思考时短回复会被思考 token 吃光 max_tokens，
导致正文为空**（实测 dots3 给 100 token 全部花在思考上）：

- **Ollama 分支**：`think=false` 参数关不掉思考，采用**预置空 `<think>` 块**让模型直接
  输出正文（实测不再输出思考 token）。
- **Anthropic 分支**：`skip_think=true` 映射为官方开关 `thinking={"type":"disabled"}`，
  实测 0.4s 直出正文（开思考 2.1s）。**未关思考时不能传 temperature**——Anthropic
  思考模式强制 temperature=1，传了会报 400。

---

## 7. 当前数据资产与指标基线（2026-08-29 验收）

### 7.1 数据资产

| 资产 | 状态 |
|---|---|
| 数据源 | 14 个注册：10 启用（arXiv cs.AI/cs.CL/cs.LG、OpenAI 新闻、量子位、LangChain/LlamaIndex/vLLM/Ollama/Transformers Releases）+ 4 禁用（anthropic-news RSS 已 404、huggingface/deepmind 当前网络超时、机器之心——保留配置待网络恢复） |
| 语料 | 103 篇文档 · 727 个检索块 · 当前快照 `20260829T123320Z` |
| 索引 | `data/index/`：chunks.jsonl（~1MB）+ vectors.json（~7.4MB，bge-m3 向量） |
| 评估 | `data/evals/` 9 份报告 + baseline.json；`data/badcases/` 4 份导出 |

### 7.2 检索指标（当前索引，12 条金标）

hit@5 = **100%**（12/12）· MRR = **0.8403**（10 条排名第一）· 关键词覆盖 91.7%

### 7.3 检索优化历程（badcase 驱动，全部有报告可复现）

| 阶段 | 检索模式 | hit@5 | MRR | 结论 |
|---|---|---|---|---|
| 初始（81 篇） | 纯词法 BM25 | 70% | 0.700 | 3 个 badcase：g004 跨语言失配、g006/g007 源缺失 |
| badcase 数据修复（103 篇） | 纯词法 BM25 | 60% | 0.395 | 短文档入库反被 BM25 长度归一化偏爱，**指标下降是真实信号** |
| 混合 v1 | BM25 + nomic-embed-text | 80% | 0.545 | 英文向量模型：中文查询"同语言扎堆" |
| 混合 v2 | BM25 + bge-m3 | 100% | 0.812 | 多语言向量模型：跨语言语义对齐 |
| **质量感知排序（当前）** | **BM25 + bge-m3 + 质量惩罚** | **100%** | **0.840** | 标题级短文档 RRF 分数降权 0.5~0.9 |

注意第二行：**数据修复后纯词法指标反而下降**——这不是事故，是这条链路的价值证明：
短文档入库改变了 BM25 的分布，门禁如实报警，推动检索层升级。

### 7.4 生成模型对比（同一金标集 × 同一检索层，`eval run -p` 切换）

| 生成模型 | 部署 | LLM 评分 | 通过率 | 单轮耗时 | 备注 |
|---|---|---|---|---|---|
| qwen3-vl:8b | 本地 Ollama | 3.33 | 83%（10/12） | ~20s | 零成本断网可用；g006/g008 judge < 3 |
| dots3-note-prev | 远程 API（Anthropic 格式） | **4.42** | **100%** | ~2s | 质量全面占优；judge JSON 不合规需正则兜底 |

两份报告的检索层指标**完全一致**（hit 100% / MRR 0.8403）——差距全部来自生成层的
指令遵循与长上下文理解。这正是多供应商抽象的意义：**换模型只改一行 toml，
同一金标集量化差距**。

### 7.5 质量门禁状态

ruff 0 错误 · mypy strict（37 文件）通过 · pytest 102 个单元测试全绿
（integration 标记默认排除）。

---

## 8. 关键工程决策与踩坑记录（接手必读）

按时间顺序，每条都是真实踩坑 + 修复方案，**不要重新踩**：

1. **judge 提示词的花括号必须转义**。`_JUDGE_PROMPT` 用 `str.format()` 填充，
   模板里的 JSON 字面量 `{"score": ...}` 会被解析成占位符抛 KeyError。
   写法是 `{{"score": ...}}`。新增含 JSON 样例的 prompt 时切记。

2. **RSS 快讯源要分级入库（min_chars）**。OpenAI RSS 只有 ~150 字快讯
   （全文页被 Cloudflare 403 拦截），量子位连摘要都没有（正文页 403 反爬）。
   一刀切的高门槛会把它们全丢掉（g006/g007 badcase 根因：**源根本没入库**）；
   一刀切的低门槛会放进垃圾。方案：按源分级——openai-news `min_chars=120`（快讯级）、
   qbitai `min_chars=5`（标题级，标题本身即有效情报），arXiv 保持 300（正文级）。

3. **短文档会污染 BM25**。标题党短文（"Agent""RAG" 等高频词）+ BM25 长度归一化
   对短文的加成，能把正确结果整体挤出 top-5（数据修复后 hit 反而从 70% 掉到 60%）。
   这不是删数据能解决的——需要向量分支稀释词法噪声（混合检索的直接动因）。
   **2026-08-30 补充**：混合检索后，标题级短文档（qbitai 平均 20 字符 search_text）
   仍靠词法分支高分挤占 top-1（如 g001 vLLM 查询被 qbitai 标题遮蔽）。
   修复方案：RRF 融合后按 search_text 长度计算质量乘数（0.5~1.0），
   对标题级文档降权。MRR 从 0.812 → 0.840。

4. **中文语料必须用多语言向量模型**。nomic-embed-text 是英文模型：中文查询与中文
   文档"同语言扎堆"（分布一致性压过语义相关性），跨语言（中文问题 ↔ 英文 arXiv 论文）
   匹配失败——这是 badcase g004 的根因。换 bge-m3（多语言）后该类查询从 miss 变 rank-1。

5. **本地 8GB 显存的上下文预算**。RTX 4070 Laptop 装不下 5.5GB 模型 + 6144 上下文的
   KV 缓存 + bge-m3：超出后 Ollama 把层卸载到内存，生成速度从 ~40 tok/s 塌到
   0.6 tok/s（30 倍劣化，单次生成 600s 超时）。方案：`num_ctx = 3072`（实测显存内
   上限）+ answer 层上下文 5000 字符上限 + skip_think 时不留思考余量。改检索块数或
   上下文长度前先想显存账。

6. **思考型模型的思考开关**（详见 §6.3）：Ollama 预置空 `<think>` 块；
   Anthropic `thinking={"type":"disabled"}`，且关思考时才能传 temperature。

7. **LLM-as-judge 解析必须容错**。dots3 实测输出 `comment` 值不带引号的非法 JSON，
   严格 `json.loads` 会让该模型的评分**整批静默丢失**（12 条只剩 6 条有分）。
   `_parse_judge_reply` 的策略：先严格 JSON（截取首尾花括号之间），
   失败再正则兜底提取 score / comment。接入任何新模型跑 eval 后，
   **先检查 entries 里 judge_score 为 null 的比例**。

8. **.env 不会自动进 os.environ**。pydantic-settings 只把 .env 填充到 Settings 字段，
   而 ProviderConfig 的 `"env:VAR"` 语法读的是 `os.environ`——多供应商模式下 Key 会
   静默读不到。`load_providers` 里的 `_bridge_env_file()` 把 .env 键值注入 os.environ
   （真实环境变量优先）。新增依赖环境变量的功能时记住这条桥已存在。

9. **数据库连接必须设 connect_timeout=5**。未配置的本地端口（pg 没启动）会让
   psycopg 无限挂起而不是报错——整个 CLI 会卡死。快速失败 + 可读错误比静默挂起好。

10. **生成失败要降级不要中断**。eval runner 里单条生成异常（超时/模型挂）只跳过该条
    生成指标，检索层照常算。同理 judge 异常返回 (None, None) 而非抛出。

---

## 9. 运行手册

### 9.1 环境准备

```powershell
# Python 3.11+，uv 管理
uv venv --python 3.11 ; .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# 本地模型（Ollama）
ollama pull qwen3-vl:8b     # 生成模型（~5.5GB）
ollama pull bge-m3          # 向量模型

# 密钥：复制 .env.example → .env，填入
#   LLMKB_DOTS3_API_KEY=...    （远程 dots3-note-prev）
#   LLMKB_DEEPSEEK_API_KEY=... （远程 deepseek-chat，可选）

# 可选：pgvector 后端（不用则跑本地文件索引，功能等价）
docker compose up -d db
```

### 9.2 典型工作流

```powershell
llm-intel-kb doctor                    # ① 环境体检（改配置后先跑这个）
llm-intel-kb data fetch                # ② 抓取数据源 → staging
llm-intel-kb data enrich               # ③ LLM 富化（缓存命中不重复付费）
llm-intel-kb data commit               # ④ 提交 → 新快照
llm-intel-kb index build               # ⑤ 切块 + 双索引
llm-intel-kb eval run -p local-ollama  # ⑥ 评估（换 -p dots3 对比远程模型）
llm-intel-kb eval gate                 # ⑦ 回归门禁（退化即退出码 1）
llm-intel-kb eval baseline             # ⑧ 指标达标后设为新基线
llm-intel-kb serve --port 8000         # ⑨ Web UI
```

日常验证：`make lint && make typecheck && make test`（或直接 ruff / mypy / pytest）。

### 9.3 接手后的改动流程（重要）

改任何检索/切块/生成逻辑后：**必须** `eval run` + `eval gate`，指标退化就修或者
有意识地重设基线（并说明理由）。**禁止**跳过门禁直接宣称"改好了"。
新发现的 badcase 走回流：复盘 → 追加 golden/qa.toml → 重跑确认。

---

## 10. 硬约束（不可违反）

1. **同时支持本地与远程模型对比**：本地 Ollama（qwen3-vl:8b）和远程 API
   （DeepSeek、dots3 等）必须都可用、可对比。
2. **模型只能通过 llm_providers.toml 配置**，不允许为接模型改代码
   （新协议类型才允许加 adapter 分支）。
3. **数据必须过质量评估门禁才能切换/上线**：gate 拦截 = 不合并。
4. **质量稳定性优先于功能数量**：不加没有评估覆盖的功能。
5. 配置统一 `LLMKB_` 前缀，优先级：环境变量 > .env > 默认值。
6. 金标集只增不改（保证历史评估可跨版本对比）。
7. Web 前端保持自包含（无 CDN 依赖，断网可用）。

---

## 11. 已知局限与可选扩展方向

接手者可以往这些方向走（均不违反硬约束）：

- **评估规模**：金标集 12 条偏小，可扩到 50+；可加检索层负例（应拒绝回答的问题）。
- **重排（rerank）**：RRF 之后接 cross-encoder 重排，预期 MRR 还有空间。
- **查询改写**：中文查询先 LLM 改写/扩展再检索，可能改善 g004 类泛化查询。
- **增量索引**：当前 index build 是全量重建；快照差分 + 增量向量化是自然演进。
- **数据源**：4 个禁用源待网络恢复；可加 Anthropic/Mistral 博客的替代抓取方式。
- **eval gate 接 CI**：门禁已支持退出码语义，接 GitHub Actions 即可。
- **成本核算**：enrich 已有缓存，可再加调用次数/token 统计面板。

---

*交接文档完。任何与 README.md 冲突之处以本文档为准（本文档更新于 2026-08-30，
对应代码状态：质量感知排序已上线，MRR 0.8403，102 测试全绿）。*
