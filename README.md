# llm-intel-kb — AI/LLM 领域情报知识库

> 数据集生命周期管线 × 工程化 RAG × 评估门禁 —— 一个把「数据可靠性」放在和「检索效果」同等地位的垂直领域知识库。

面向 AI 应用工程师 / AI 产品方向的**作品集项目**。它回答一个真实问题：
**当你的知识库数据每天都在变（新论文、新版本、新资讯），你怎么保证更新之后系统没有悄悄变坏？**

答案是一条完整的工程链路：聚合公开数据源 → 版本化入库 → 切块建索引 → 混合检索生成 →
**用金标问答集做自动评估，回归门禁不通过就不切换索引**。

## 一图看懂

```
                    ┌───────────────────── 数据面（M1）─────────────────────┐
  12 个数据源        暂存区          内容哈希去重        快照（版本化）
  arXiv ×3 ──fetch──▶ staging ──commit──▶ corpus(不可变) ──▶ snapshot_2
  OpenAI/量子位      （可丢弃）        清洗+语种检测            ↑ parent
  GitHub Releases ×5                                            snapshot_1
                    └────────── LLM 富化（中文摘要+英文主题词，缓存命中不重复付费）──┘

                    ┌───────────────────── 服务面（M2）─────────────────────┐
  snapshot ──chunk──▶ 结构感知切块（段落边界+标题前缀+相邻重叠）
                    ├──▶ BM25 词法索引（纯 Python，中英混合分词）
                    └──▶ 向量索引（本地文件 / pgvector HNSW 二选一）
                              └──search──▶ RRF 融合排序 ──ask──▶ 引用生成 [n]

                    ┌───────────────────── 评估面（M3）─────────────────────┐
  golden/qa.toml ──eval run──▶ hit@5 / MRR / 关键词覆盖 / LLM-as-judge
                    └──eval gate──▶ 与基线对比，指标退化即拒绝（退出码 1）
                    └──badcase 回流──▶ 人工复盘后追加进金标集

                    ┌───────────────────── 交付（M4）───────────────────────┐
  CLI（typer + rich）× Web UI（FastAPI + 自包含单页，无 CDN、断网可用）
```

## 里程碑与验收状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 骨架 · 配置 · doctor 体检 · 多模型供应商 · compare 对比 · CI | ✅ 完成 |
| M1 | 数据面 CLI：源发现 → 清洗富化 → 版本化存储 → 增量更新 | ✅ 完成 |
| M2 | 服务面：结构感知切块 → 混合检索（BM25+向量 RRF）→ 引用生成 | ✅ 完成 |
| M2+ | pgvector 可选后端（HNSW ANN，与本地文件索引接口对齐） | ✅ 完成 |
| M3 | 评估面：金标集 → 自动评估 → 回归门禁 → badcase 回流 | ✅ 完成 |
| M4 | 交付：Web UI（问答/语料/版本/评估）· 作品集文档 | ✅ 完成 |
| M5 | 证据级评估（块级证据率/忠实度）· 领域守卫（规则+豁免+语义兜底+回归闭环） | ✅ 完成 |

**真实数据验收**（2026-08-29，混合检索 BM25 + bge-m3；生成层对比本地 qwen3-vl:8b 与远程 dots3-note-prev）：

| 指标 | 数值 |
|---|---|
| 数据源 | 12 个注册（arXiv cs.AI/cs.CL/cs.LG、OpenAI 新闻、量子位、LangChain/LlamaIndex/vLLM/Ollama/Transformers Releases 等） |
| 语料规模 | 103 篇文档 · 727 个检索块 · 4 个版本快照 |
| 检索评估 | 12 条金标（初期）：hit@5 = 100% · MRR = 0.840；**100 条金标（当前基线）：hit@5 = 78% · MRR = 0.527**（混合检索：BM25 + bge-m3 向量 RRF 融合 + 内容质量感知排序） |
| 生成评估 | 同一检索层换生成模型（12 条金标）：本地 8B judge 3.33 / 通过率 83% → 远程 dots3 judge 4.42 / 通过率 100%；100 条金标下 dots3 judge 3.0 / 通过率 55%（含 negative 反例与守卫拦截场景） |
| 问答链路 | 端到端 10.7s（本地 8B 模型），答案带 `[n]` 引用标注，引用率 100% |
| 质量门禁 | ruff 0 错误 · mypy strict 全量通过 · 146 个单元测试全绿 |

### 生成模型对比（同一金标集 × 不同生成模型，eval run -p 切换）

| 生成模型 | 部署 | LLM 评分 | 通过率 | 单轮耗时 | 特点 |
|---|---|---|---|---|---|
| qwen3-vl:8b | 本地 Ollama | 3.33 | 83%（10/12）| ~20s | 零成本断网可用；长资料理解偶尔跑偏（g006 答非所问）|
| dots3-note-prev | 远程 API（Anthropic Messages 格式）| **4.42** | **100%（12/12）** | ~2s | 生成/判分质量全面占优；judge 输出 JSON 不合规（comment 不带引号），解析层需正则兜底 |

检索层指标（hit@5 / MRR / 关键词覆盖）只取决于索引，两个模型完全一致——
这说明差距全部来自**生成层的指令遵循与长上下文理解**，而检索基座是稳的。
这也是本项目多供应商抽象的意义：**换模型只改一行 toml，同一金标集量化差距**。

### 检索优化历程（badcase 驱动的迭代，全部数据可复现）

| 阶段 | 检索模式 | hit@5 | MRR | 说明 |
|---|---|---|---|---|
| 初始（81 篇） | 纯词法 BM25 | 70% | 0.700 | 3 个 badcase：g004 跨语言失配、g006/g007 源缺失 |
| badcase 数据修复（103 篇） | 纯词法 BM25 | 60% | 0.395 | 快讯级短文档入库后 BM25 被标题党污染（长度归一化偏爱短文） |
| 混合检索 v1 | BM25 + nomic-embed-text | 80% | 0.545 | 英文向量模型：中文查询"同语言扎堆"匹配中文标题，而非语义相关文档 |
| 混合检索 v2 | BM25 + bge-m3 | 100% | 0.812 | 多语言向量模型：跨语言语义对齐，12/12 命中、9 条排名第一 |
| **质量感知排序（当前）** | **BM25 + bge-m3 + 质量惩罚** | **100%** | **0.840** | 对标题级短文档（<100字符）RRF 分数降权 0.5~0.9，核心内容排位提升 |

badcase 复盘沉淀的三条工程结论（详见 `data_sources.toml` 与 `llm_providers.toml` 注释）：

1. **RSS 快讯源要分级入库**：OpenAI RSS 只带 150 字快讯、量子位连摘要都为空（全文页均被反爬拦截）。
   按「快讯级（min_chars=120）/ 标题级（min_chars=5）」分级放行，标题本身就是有效情报。
2. **短文档会污染 BM25**：标题党关键词（"Agent"“RAG”）+ 极短文本的长度归一化加成，能把正确结果
   整体挤出 top-5。这不是删数据能解决的——需要向量分支稀释词法噪声。
3. **中文语料必须用多语言向量模型**：nomic-embed-text 是英文模型，中文查询会与中文文档"同语言
   扎堆"（分布一致性压过语义相关性）。换 bge-m3 后跨语言检索（中文问题 ↔ 英文 arXiv 论文）
   直接从 miss 变 rank-1。

## 快速开始

> 📄 文档导航：现场演示与面试问答看 [docs/面试备战手册.md](docs/面试备战手册.md)；
> 完整使用说明看 [USAGE.md](USAGE.md)；架构决策与历史踩坑看 [HANDOFF.md](HANDOFF.md)。
> 历史设计文档归档在 `docs/archive/`。

前置：Python 3.11+；模型服务（本地 Ollama 或远程 API 至少其一）；数据库可选（不用 pgvector 就跑本地文件索引）。

```powershell
# 1. 安装（uv 或 pip 二选一）
uv venv --python 3.12 && uv pip install -e ".[dev]"
#   pip 方式：python -m venv .venv && .venv\Scripts\pip install -e ".[dev]"

# 2. 配置：本地 Ollama 无需任何 Key；远程 API 才需要
Copy-Item .env.example .env      # 编辑填入 LLMKB_DEEPSEEK_API_KEY（可选）

# 3. 体检（数据库 / pgvector / 所有模型供应商连通性）
.venv\Scripts\llm-intel-kb.exe doctor

# 4. 一条命令看多模型对比
.venv\Scripts\llm-intel-kb.exe compare "用一句话解释什么是RAG"
```

### 跑通完整链路（5 条命令）

```powershell
llm-intel-kb data fetch            # 抓取 12 个源到暂存区
llm-intel-kb data commit           # 清洗入库（哈希去重）+ 创建快照
llm-intel-kb data enrich -l 20     # LLM 富化前 20 篇（缓存命中不重复付费）
llm-intel-kb index build           # 切块 + 建索引（无 embedding 供应商则纯词法）
llm-intel-kb eval run              # 金标集评估 → 生成报告与 badcase
```

然后体验两种交付形态：

```powershell
llm-intel-kb index search "vllm 性能优化"        # 纯检索（不耗生成 token）
llm-intel-kb index ask "vLLM 最近有哪些版本更新？"
llm-intel-kb serve                              # 浏览器打开 http://127.0.0.1:8000
```

Web UI 提供：**智能问答**（带引用来源）· **语料浏览**（分页/搜索/富化摘要）·
**版本历史**（快照对比/一键回滚）· **评估报告**（指标/逐题命中/badcase）。

## 多模型供应商架构

本地模型（Ollama / LM Studio / vLLM）与远程 API（DeepSeek / OpenAI / SiliconFlow）统一为
一套供应商抽象，在 [llm_providers.toml](llm_providers.toml) 声明，Key 走环境变量注入：

```toml
[deepseek]
kind = "chat"                      # chat / embedding / both
base_url = "https://api.deepseek.com"
api_key = "env:LLMKB_DEEPSEEK_API_KEY"
model = "deepseek-chat"

[local-ollama]
kind = "chat"
adapter = "ollama"                 # 原生 /api/chat 接口（思考型模型更稳）
base_url = "http://localhost:11434/v1"
model = "qwen3-vl:8b"
skip_think = true                  # 预置空思考块：qwen3 不再输出思考 token
num_ctx = 3072                     # 8GB 显存实测上限；6144 会触发显存溢出降速

[dots3]                            # Anthropic Messages 格式的远程 API
kind = "chat"
adapter = "anthropic"              # /v1/messages：system 走顶层字段、x-api-key 认证
base_url = "https://note3-prev-api.askdiandian.com"
api_key = "env:LLMKB_DOTS3_API_KEY"
model = "dots3-note-prev"
skip_think = true                  # 映射为官方开关 thinking={"type":"disabled"}
```

**适配器三分支**：`adapter = "openai"`（默认，OpenAI 兼容接口）、`adapter = "ollama"`
（原生接口）、`adapter = "anthropic"`（Messages API）。三者共享同一套供应商抽象与
`skip_think` 语义（跳过思考直接输出正文），但实现各异：qwen3 靠预置空 `<think>` 块、
dots3 靠官方 thinking 开关——这是适配层吸收模型差异的典型样本。思考型模型不关思考的
代价：思考 token 耗尽 `max_tokens` 导致正文为空、耗时数倍于正文。
不用 toml 时回退 `.env` 单一供应商模式，兼容最小部署。

**.env 桥接**（真实踩坑）：pydantic-settings 只把 `.env` 填充进 Settings 字段、
不写进程环境，而供应商的 `env:VAR` 语法读 `os.environ`——不桥接的话 `.env` 里的
API Key 对多供应商模式永远不可见（doctor 全部 SKIP）。修复：加载供应商时把
`.env` 键值注入 `os.environ`（真实环境变量优先）。

**显存预算是本地 RAG 的隐形天花板**（真实踩坑，排查 40 分钟）：8GB 笔记本 GPU 上
`qwen3-vl:8b`（5.5GB）+ `bge-m3`（0.7GB）+ `num_ctx=6144` 的 KV 缓存超出显存，
Ollama 静默把层卸载到内存——生成速度从 ~40 tok/s 塌到 0.6 tok/s，单次生成 600 秒超时，
且无任何错误提示。修复三件套：`num_ctx` 降到 3072、`skip_think` 开启时去掉思考
token 余量（`num_predict` 从 5120 收紧到实际答案预算）、生成上下文上限 9000→5000 字符。
同一段代码修复前后：600s 超时 → 11-22s 正常返回。

## 混合检索：词法 + 向量，RRF 融合

`index build` 时若声明了 `kind = "embedding"` 的供应商（本地或远程均可），检索自动从
纯词法升级为混合模式：BM25 与向量两路各自排序，RRF（倒数排名融合）合并——任何一路
进入 `--top` 前列的结果都会胜出，词法噪声被语义信号稀释，反之亦然。

```toml
# llm_providers.toml：一行声明即启用（本地 Ollama 零成本）
[local-ollama-embedding]
kind = "embedding"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
model = "bge-m3"          # 先执行：ollama pull bge-m3
```

```powershell
ollama pull bge-m3                  # 1.2 GB，多语言向量模型
llm-intel-kb index build            # 重新切块 + 向量化（无 embedding 供应商则纯词法）
llm-intel-kb index search "推理优化"  # 返回结果带 mode 与 rrf_score 字段
```

**为什么是 bge-m3 而不是 nomic-embed-text**：语料是中英混合（arXiv 英文论文 + 中文资讯），
检索必须跨语言。英文向量模型下中文查询会与中文文档「同语言扎堆」——分布一致性压过
语义相关性，排名前列全是中文标题党。bge-m3 做中英语义对齐后，badcase g004（中文问题 ↔
英文论文）从 miss 直接变 rank-1。切换模型只需改 toml 一行 + 重建索引，检索模式、指标
对比全部落盘可查（见下文优化历程表）。

## 索引后端：本地文件 or pgvector

```powershell
llm-intel-kb index build --backend local   # 默认：JSONL + 向量文件，零依赖
llm-intel-kb index build --backend pg      # pgvector：HNSW ANN 检索，规模化
llm-intel-kb index build --backend auto    # 探测 DB 可用性自动选择
docker compose up -d db                    # pgvector/pgvector:pg16 一键起库
```

两个后端接口完全对齐（`index_meta` / `index_chunks` / `index_vectors` 三表 vs 本地三文件），
加载层 [loader.py](src/llm_intel_kb/index/loader.py) 读 meta 标记透明切换，
DB 不可用时给出可操作的降级提示（重启数据库或 `--backend local` 重建）。

## 评估与回归门禁

金标集 [golden/qa.toml](golden/qa.toml) 维护贴近真实问法的问题，
`eval run` 输出四层指标：

- **检索层**：hit@k（期望源是否进 top-k）、MRR（期望源排多前）
- **生成层**：关键词覆盖率、引用率、LLM-as-judge 评分（无 judge 模型时自动跳过）
- **端到端**：逐题记录命中排名 / 检索来源 / 回答全文，badcase 落盘 `data/badcases/*.jsonl`

```powershell
llm-intel-kb eval run                  # 跑评估，生成报告
llm-intel-kb eval run -p dots3         # 换生成模型跑同一金标集（多模型对比）
llm-intel-kb eval gate                 # 与基线对比：hit/MRR 退化 → 退出码 1
llm-intel-kb eval baseline             # 把当前报告固化为新基线
```

**judge 解析要容错**（真实踩坑）：LLM-as-judge 要求模型输出 JSON，但不同模型遵守度
不一——dots3 实测会输出 `comment` 值不带引号的非法 JSON，严格 `json.loads` 会让
该模型的评分整批静默丢失（12 条只剩 6 条有分）。解析层先严格 JSON、失败再正则兜底
提取 score/comment。Web UI 的评估页会按生成模型分组展示各模型最新指标对比。

**badcase 回流**：badcase 人工复盘后确认是「应答对却没答对」的，追加进金标集——
评估集随系统一起成长，这正是「数据集生命周期」的最后一环。

## 命令速查

| 命令 | 说明 |
|---|---|
| `llm-intel-kb version` / `doctor` | 版本 / 环境体检（存在 FAIL 退出码 1，可挂 CI） |
| `llm-intel-kb compare "问题"` | 同一问题并发发给所有就绪模型，并排对比回答与耗时 |
| `llm-intel-kb data sources / fetch / status / commit` | 源注册表 / 抓取 / 暂存区状态 / 入库建快照 |
| `llm-intel-kb data enrich -p <供应商> -l <篇数>` | LLM 富化（`--limit` 控制花费） |
| `llm-intel-kb data log / rollback <快照ID>` | 版本历史 / 回滚到指定快照 |
| `llm-intel-kb index build --backend local/pg/auto` | 切块建索引 |
| `llm-intel-kb index search "词"` | 纯检索（免费，不耗生成 token） |
| `llm-intel-kb index ask "问题" -p <供应商>` | 检索 + 引用生成 |
| `llm-intel-kb eval run / gate / baseline` | 评估 / 回归门禁 / 固化基线 |
| `llm-intel-kb serve --port 8000` | 启动 Web UI |
| `make lint / test / test-integration` | ruff + mypy strict / 单测 / 集成测试 |

## 测试与质量门禁

- **单元测试（146 个）**：管线（清洗/哈希/去重/快照/回滚）、检索（BM25/RRF/切块/质量惩罚）、
  评估（指标计算/门禁判定/块级证据率）、守卫（规则/豁免/语义兜底/回归）、供应商解析（toml/env 回退/`env:` Key 语法/本地远程判定）、
  客户端 mock、CLI 渲染
- **集成测试**（`-m integration`）：真实 PostgreSQL/pgvector 幂等性、本地 Ollama 连通性
  （`LLMKB_SKIP_LOCAL_LLM=1` 可跳）、远程 API（配 Key 才跑）
- **静态检查**：ruff（E/F/I/UP/B/SIM/RUF 规则集）+ mypy strict 全量通过
- **CI**：lint → 单测矩阵（3.11/3.12）→ pgvector 容器上跑集成测试

```powershell
make lint && make test
# 集成测试环境变量：
$env:LLMKB_TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/llm_intel_kb"
pytest -m integration
```

## 配置项

全部 `LLMKB_` 前缀环境变量或 `.env`，见 [.env.example](.env.example)：

| 变量 | 说明 |
|---|---|
| `LLMKB_DATABASE_URL` | PostgreSQL（pgvector）连接串，本地文件索引时可不配 |
| `LLMKB_DEEPSEEK_API_KEY` / `LLMKB_SILICONFLOW_API_KEY` | 远程模型 Key（toml 中 `env:` 引用） |
| `LLMKB_PROVIDERS_FILE` | 多供应商配置路径（默认 `./llm_providers.toml`） |
| `LLMKB_DATA_DIR` | 语料 / 快照 / 索引 / 评估产物的本地目录（默认 `./data`） |

## 项目结构

```
llm-intel-kb/
├── src/llm_intel_kb/
│   ├── pipeline/          # 数据面：sources 抓取 · clean 清洗哈希 · commit 入库
│   │   │                  #         store 版本化存储 · enrich LLM 富化（带缓存）
│   ├── index/             # 服务面：chunking 结构切块 · lexical BM25 · embeddings
│   │   │                  #         search RRF 混合检索 · store_local / store_pg 双后端
│   ├── rag/               # 引用生成（提示词要求 [n] 标注，只依据检索结果）
│   ├── eval/              # 评估面：golden 金标集 · runner 指标 · gate 回归门禁
│   ├── guard/             # 领域守卫：规则词表 + 领域词豁免 · 语义兜底 · 回归用例
│   ├── llm/               # 供应商抽象 + OpenAI/Ollama/Anthropic 三适配器客户端
│   ├── web/               # FastAPI 后端 + 自包含单页前端（无 CDN，断网可用）
│   ├── cli/               # typer 命令：data / index / eval / guard / serve / doctor / compare
│   ├── store/db.py        # SQLAlchemy 引擎 + pgvector 扩展幂等安装
│   └── checks.py          # doctor 体检项
├── tests/                 # unit（146 个）+ integration（DB/本地模型/API）
├── golden/                # qa.toml 金标问答集（100 条）· guard_cases.toml 守卫回归（12 条）
├── docs/                  # 面试备战手册.md（演示流程+话术+问答）· archive/（历史设计文档）
├── data_sources.toml      # 数据源注册表（12 源，enabled 开关）
├── llm_providers.toml     # 模型供应商注册表
├── USAGE.md               # 完整使用说明（从启动到日常使用）
├── HANDOFF.md             # 架构决策与历史踩坑交接
└── docker-compose.yml     # pgvector:pg16
```

## 工程决策速览（作品集叙事）

1. **内容哈希寻址，语料只增不删**——文档 ID = hash(标题+正文)，天然去重、不可变、可回溯；
   抓取失败的源不视为删除，防止网络抖动误删数据。
2. **快照而非覆盖**——每次 commit 生成新快照（父指针链），增量 diff 与回滚是数据结构自带的，
   不是事后补的功能。
3. **无 embedding 供应商时自动降级纯词法检索**——系统在「零配置」下依然完整可用，
   混合检索是升级项而非前置依赖。
4. **回归门禁挡在数据与索引之间**——`eval gate` 退出码 1 即拒绝发布，
   把「数据变更可能降低检索质量」从事故变成 CI 失败。
5. **双索引后端接口对齐**——本地文件零依赖跑通全链路，pgvector HNSW 支撑规模化，
   切换成本是一行 `--backend`。
6. **思考型模型的工程化处理**——`skip_think` 预置空思考块 + `num_ctx` 放大窗口，
   解决 qwen3 空回复与 3 倍耗时，配置即生效不改代码。
7. **自包含前端**——单 HTML 无 CDN 依赖，断网、内网、离线演示均可直接打开。
8. **内容质量感知排序**——对 BM25+向量 RRF 融合后的分数按文档信息量降权，
   标题级短文档（<100字符）乘以 0.5~0.9 系数，MRR 从 0.812 提升至 0.840。

## License

个人作品集项目，用于求职展示与学习交流。
