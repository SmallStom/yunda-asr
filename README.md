# 轨道交通领域 ASR 后处理文本纠错系统

面向铁路调度场景的专业 ASR 后处理纠错系统，通过四级流水线将通用 ASR 输出的口语化、误识别文本转换为标准铁路调度用语。

## 系统架构

```
ASR原始输出
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 1: 文本预处理层 (Preprocessor) │  逆文本规范化(ITN) + 标点补全
├─────────────────────────────────────┤
│  Layer 2: 词典强制纠错 (Dictionary)   │  Trie树 + 别名映射 + 正则模式
├─────────────────────────────────────┤
│  Layer 3: 上下文感知纠错 (Context)    │  N-gram语言模型 + 共现规则 + 拼音混淆
├─────────────────────────────────────┤
│  Layer 4: 语义精修 (Semantic)        │  LLM + 领域知识检索 + 术语工具 + 实体校验
└─────────────────────────────────────┘
    │
    ▼
标准调度文本
```

### 各层职责

| 层级 | 模块 | 职责 | 技术方案 |
|------|------|------|----------|
| Layer 1 | `preprocessor.py` | 数字/时间/速度/公里标规范化，标点补全 | 正则 + cn2an |
| Layer 2 | `dictionary_corrector.py` | 同音/近音别名替换，术语标准化 | Trie树 + 别名映射 + 正则模式匹配 |
| Layer 3 | `context_corrector.py` | 上下文消歧，补充纠错 | N-gram(bigram) + 术语共现规则 + 拼音混淆候选 |
| Layer 4 | `semantic_refiner.py` / `rag_refiner.py` / `harness_refiner.py` | 跨句段语义一致性，领域知识增强纠错 | 本地LLM + RAG检索 + 术语工具 + 实体安全校验 |

### Layer 4 语义精修模式

Layer 4 支持三种模式，通过 `semantic_mode` 参数选择；同时支持通过 `LLM_PROMPT_VERSION` 切换 Prompt 版本（v1/v2）：

| 模式 | 模块 | 原理 | CER (v1) | CER (v2) | 延迟 (v2) | 适用场景 |
|------|------|------|----------|----------|-----------|----------|
| `baseline` | `semantic_refiner.py` | 单次LLM调用 + 实体校验 | 0.1656 | 0.1652 | ~0.5s | 延迟敏感 |
| `rag` | `rag_refiner.py` | 领域知识检索 + 术语工具预检索 + 单次LLM | 0.1558 | **0.0852** | ~1.7s | **效果最优** |
| `harness` | `harness_refiner.py` | 多策略竞争(基线+RAG) + 裁判LLM选择 | 0.1569 | 0.1385 | ~2.6s | 多策略兜底 |

**Prompt v2** 是铁路行车作业术语规范化专家提示词，包含角色定义、术语体系、6条推理铁律、7类场景分类和15个Few-shot示例。在RAG模式下效果提升显著：CER从0.1558降至0.0852（降幅45%）。

#### RAG 模式工作流程

```
规则层输出
    │
    ├─① 领域知识检索 (KnowledgeRetriever)
    │     从 aliases / railway_terms / asr_error_pairs 检索术语映射、历史错误模式、场景规则
    │
    ├─② 术语工具预检索 (TermTool)
    │     jieba分词 → 对每个2-4字词查询拼音相似术语 → 标注置信度(高/中/低)
    │
    ├─③ 拼接Prompt = 领域知识 + 工具hint + 原文 + 规则结果 + 修改历史
    │
    ├─④ 单次LLM调用
    │
    ├─⑤ 实体校验 (EntityGuard) — 拦截数字篡改
    │
    └─ 输出纠错结果
```

#### Harness 模式工作流程

```
规则层输出
    │
    ├─ 策略A: 基线LLM纠错 (SemanticRefiner)
    ├─ 策略B: RAG+术语工具纠错 (RAGRefiner)
    │
    ├─ 如果A==B → 直接返回
    ├─ 如果A≠B → 裁判LLM选择更优结果
    │
    ├─ 实体校验
    └─ 输出
```

## 快速开始

### 环境要求

- Python >= 3.9
- 依赖库：jieba, pypinyin, cn2an, openai, rapidfuzz, pandas, openpyxl

### 安装

```bash
cd asr
pip install -e .
```

### 基础用法

```python
from src.pipeline import PostCorrectionPipeline

pipeline = PostCorrectionPipeline()
pipeline.warmup()  # 预热，避免首次调用冷启动

# 基础纠错（Layer 1-3）
result = pipeline.run("十八号道差开通反位，信号好了")
print(result.corrected)
# 输出: 18号道岔开通反位，信号好了。

# 查看各层输出
for layer, text in result.layer_outputs.items():
    print(f"{layer}: {text}")

# 查看修改详情
for detail in result.details:
    print(f"[{detail.layer}] {len(detail.changes)} 处修改")

# 启用 Layer 4 语义精修 — RAG模式
result = pipeline.run("点击送人节按钮", layers=[1, 2, 3], enable_semantic=True, semantic_mode="rag")
print(result.corrected)
# 输出: 点击总人解按钮。

# 启用 Layer 4 语义精修 — RAG + v2 Prompt（当前最优效果）
import os
os.environ["LLM_PROMPT_VERSION"] = "v2"
pipeline = PostCorrectionPipeline()  # 切换prompt后需重新实例化
result = pipeline.run("点击送人节按钮", layers=[1, 2, 3], enable_semantic=True, semantic_mode="rag")
print(result.corrected)

# 启用 Layer 4 语义精修 — Harness模式
result = pipeline.run("点击送人节按钮", layers=[1, 2, 3], enable_semantic=True, semantic_mode="harness")
print(result.corrected)
```

## 交互式演示

### Gradio Web演示（推荐）

提供可视化界面，支持文本/音频输入、热词编辑、分步结果展示、Prompt查看。

```bash
# 安装依赖
pip install gradio pandas openpyxl

# 启动服务
python scripts/gradio_demo.py
```

访问 `http://localhost:7860`，界面包含：
- **左侧**：输入方式选择、ASR来源选择（本地ASR(VibeVoice) / ElevenLabs ASR）、ASR配置、热词编辑、纠错模式选择、Prompt版本选择、正确答案（可选）、各层方法说明
- **右侧**：运行状态、Step 0-4 分步改写结果（带修改高亮）、最终输出（相对原文总修改高亮）、正确答案对比、当前大模型Prompt展示
- **底部**：快速示例，一键加载典型case

**流式展示**：点击"开始纠错"后，每完成一层会实时更新右侧结果，无需等待全部完成。绿色高亮表示新增/修改内容，红色删除线表示被删除/替换内容。

#### ASR来源配置

演示支持两种ASR来源，通过项目根目录的 `.env` 文件配置：

```bash
# 1. 复制示例文件
cp .env.example .env

# 2. 编辑 .env，填入你的ASR服务地址和密钥
```

`.env` 示例内容：

```ini
# LLM 配置
LLM_BASE_URL=http://192.168.1.119:8012/v1
LLM_MODEL=Qwen3.6-27B
LLM_API_KEY=dummy-key-for-local
LLM_PROMPT_VERSION=v2

# 本地ASR（VibeVoice格式，base_url不要带/v1后缀）
LOCAL_ASR_BASE_URL=http://192.168.1.119:8015
LOCAL_ASR_MODEL=vibevoice
LOCAL_ASR_API_KEY=dummy-key-for-local

# ElevenLabs ASR（https://elevenlabs.io/app/api/api-keys）
ELEVENLABS_ASR_MODEL=scribe_v1
ELEVENLABS_ASR_API_KEY=sk_your_elevenlabs_api_key_here
```

注意：ElevenLabs STT API 目前没有官方热词参数，界面中的热词对 ElevenLabs ASR 不生效，但会用于后处理纠错层。

`.env` 文件已加入 `.gitignore`，不会被提交到代码仓库。

### 命令行演示

当Gradio无法启动时使用：

```bash
python scripts/console_demo.py
```

按提示输入ASR文本、选择模式（rag/harness/baseline）、选择Prompt版本（v1/v2）、输入热词，即可查看分步结果和Prompt。

## API 服务部署

项目已提供 FastAPI 服务，可直接以 API 形式接入业务系统。

### 启动 API 服务

```bash
pip install -e .
python scripts/start_api.py
```

访问 API 文档：`http://localhost:8000/docs`

### 核心接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/ready` | GET | 就绪检查 |
| `/api/v1/correct` | POST | 单条文本纠错 |
| `/api/v1/correct/batch` | POST | 批量文本纠错 |
| `/api/v1/hotwords` | GET/POST/PUT/DELETE | 热词管理 |
| `/api/v1/prompts` | GET/POST/PUT | Prompt 版本管理 |
| `/api/v1/dify-sync/*` | POST | Dify 同步（可选） |
| `/api/v1/metrics` | GET | 服务指标 |

### 调用示例

```bash
curl -X POST "http://localhost:8000/api/v1/correct" \
  -H "Content-Type: application/json" \
  -d '{"text": "十八号道差开通反位，信号好了", "layers": [1, 2, 3]}'
```

### Docker 部署

```bash
docker-compose up -d --build
```

详细部署说明见 [docs/deployment.md](docs/deployment.md)，Dify 集成说明见 [docs/dify_integration.md](docs/dify_integration.md)。

### 指定层号

```python
# 仅使用 Layer 1 + Layer 2
result = pipeline.run(text, layers=[1, 2])

# 仅使用 Layer 2
result = pipeline.run(text, layers=[2])

# 全部基础层 + 语义精修
result = pipeline.run(text, layers=[1, 2, 3], enable_semantic=True, semantic_mode="rag")
```

### 批量处理

```python
texts = ["十八号道差开通反位", "新号机灯光熄灭", ...]
results = pipeline.run_batch(texts, layers=[1, 2, 3])
```

## LLM 配置

Layer 4 语义精修需要 LLM 服务。默认连接 `192.168.1.119:8012` 的 Qwen3.6-27B 模型。

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_BASE_URL` | `http://192.168.1.119:8012/v1` | LLM API 地址 |
| `LLM_MODEL` | `Qwen3.6-27B` | 模型名称 |
| `LLM_API_KEY` | `dummy-key-for-local` | API 密钥（本地部署通常无鉴权） |
| `LLM_PROMPT_VERSION` | `v1` | Prompt 版本 |

```bash
# 自定义 LLM 地址
export LLM_BASE_URL=http://your-server:port/v1
export LLM_MODEL=your-model-name
```

### 降级策略

- LLM 调用失败 → 自动回退到 Layer 3 结果
- 实体校验失败（数字被篡改） → 回退到 Layer 3 结果
- 超时（30s）或连接错误 → 指数退避重试（最多 3 次），最终降级

## 项目结构

```
asr/
├── src/                              # 核心源码
│   ├── __init__.py
│   ├── pipeline.py                   # 四级流水线编排器
│   ├── preprocessor.py               # Layer 1: 文本预处理
│   ├── itn_rules.py                  # ITN 规则定义
│   ├── dictionary_corrector.py       # Layer 2: 词典纠错
│   ├── context_corrector.py          # Layer 3: 上下文感知纠错
│   ├── ngram_model.py                # N-gram 语言模型
│   ├── collocation_rules.py          # 术语共现规则
│   ├── phonetic_candidate.py         # 拼音混淆候选生成
│   ├── semantic_refiner.py           # Layer 4 基线: 单次LLM语义精修
│   ├── rag_refiner.py                # Layer 4 RAG: 领域知识检索增强纠错
│   ├── harness_refiner.py            # Layer 4 Harness: 多策略竞争+裁判
│   ├── term_tool.py                  # 术语查询工具（拼音相似术语检索）
│   ├── knowledge_retriever.py        # 领域知识检索引擎
│   ├── llm_client.py                 # LLM 客户端封装
│   ├── entity_guard.py               # 实体安全校验
│   ├── feedback_collector.py         # 错误案例收集
│   └── prompts/                      # Prompt 版本管理
│       ├── registry.json
│       ├── v1/
│       │   ├── system.txt
│       │   └── user_template.txt
│       └── v2/
│           ├── system.txt            # 铁路专家提示词（含术语体系、铁律、场景、Few-shot）
│           └── user_template.txt
├── data/                             # 数据文件
│   ├── lexicon/
│   │   ├── aliases.json              # 别名映射 (1384条)
│   │   ├── railway_terms.json        # 术语库 (882条)
│   │   ├── word_confusion.json       # 词语混淆映射 (42条，配置化)
│   │   ├── phrase_patterns.json      # 短语级纠错模式 (12条，配置化)
│   │   ├── phonetic_confusion.json   # 拼音混淆规则
│   │   ├── hotwords.json             # 热词列表 (874条)
│   │   ├── hotwords.txt / .csv       # 热词列表（其他格式）
│   │   └── new_terms_raw.json        # 原始新术语数据
│   ├── corpus/
│   │   ├── railway_corpus.txt        # 铁路领域语料
│   │   ├── railway_corpus_raw.txt    # 原始语料
│   │   ├── ngram_model.json          # 预训练 N-gram 模型
│   │   └── asr_error_pairs.jsonl     # ASR错误对（RAG知识源）
│   ├── asr_testset/
│   │   ├── asr_test_pairs.jsonl      # 标准测试对
│   │   ├── asr_test_pairs_elevenlabs.jsonl  # ElevenLabs ASR测试集(100条)
│   │   ├── asr_test_pairs_long.jsonl       # 长文本测试对
│   │   └── asr_test_pairs_long_hotwords.jsonl  # 长文本+热词测试对
│   └── feedback/                     # 错误案例存储（运行时生成）
├── scripts/                          # 工具脚本
│   ├── build_corpus.py               # 语料构建
│   ├── build_lexicon.py              # 词典构建
│   ├── build_asr_testset.py          # ASR 测试集构建
│   ├── evaluate_on_testset.py        # 批量评估(Layer 1-3)
│   ├── evaluate_three_directions.py  # 多模式评估(Layer 4 对比)
│   ├── compare_prompt_versions.py    # Prompt版本效果对比
│   ├── merge_prompt_version_reports.py # 合并v1/v2报告
│   ├── gradio_demo.py                # Gradio Web演示
│   ├── console_demo.py               # 命令行交互演示
│   ├── retrain_ngram.py              # N-gram 重训练
│   ├── update_lexicon.py             # 词典增量更新
│   ├── augment_corpus.py             # 语料扩增清洗
│   ├── generate_hotwords.py          # 热词生成
│   ├── analyze_error_patterns.py     # 错误模式分析
│   ├── clean_aliases.py              # 别名清理工具
│   └── clean_dangerous_aliases.py    # 危险别名清理工具
├── tests/                            # 测试
│   ├── test_preprocessor.py          # Layer 1 测试
│   ├── test_itn_rules.py             # ITN 规则测试
│   ├── test_dictionary.py            # Layer 2 测试
│   ├── test_context.py               # Layer 3 测试
│   ├── test_semantic.py              # Layer 4 测试
│   ├── test_pipeline.py              # 流水线集成测试
│   ├── test_e2e.py                   # 端到端测试
│   ├── test_e2e_long.py              # 长文本端到端测试
│   ├── test_llm_correction.py        # LLM纠错测试
│   ├── test_three_directions.py      # 多模式对比测试
│   ├── benchmark.py                  # 性能基准测试
│   └── utils/
│       └── metrics.py                # 评估指标工具
├── pyproject.toml                    # 项目配置
└── reports/                          # 评估报告（运行时生成）
```

## 评估

### 多模式对比评估

```bash
# 评估 baseline / rag / harness 三种模式，输出Excel报告
python scripts/evaluate_three_directions.py --testset data/asr_testset/asr_test_pairs_elevenlabs.jsonl
```

输出Excel报告包含：
- Sheet1 汇总指标：各模式CER、改善/劣化数、耗时
- Sheet2 逐条对比：ASR原始、正确文本、规则纠错、各模式LLM结果（条件格式高亮改善/劣化）

### Layer 1-3 评估

```bash
# 评估 Layer 1-3
python scripts/evaluate_on_testset.py --layers 1 2 3 --output-dir reports/

# 启用语义精修评估
python scripts/evaluate_on_testset.py --enable-semantic

# 限制样本数
python scripts/evaluate_on_testset.py --limit 100
```

### 评估指标

| 指标 | 说明 |
|------|------|
| CER (字符错误率) | 编辑距离 / max(长度)，越低越好 |
| 术语准确率 | 标准术语命中数 / 参考术语总数 |
| 实体保真率 | 数字实体（车次/道岔/股道等）保真比例 |
| 各层触发率 | 各层实际产生修改的样本比例 |
| 改善/不变/劣化计数 | 纠错后 CER 变化分布 |

### 最新评估结果（ElevenLabs 100条测试集）

#### Prompt v1 结果

| 模式 | 原始CER | 规则CER | LLM CER | 改善 | 劣化 | 耗时 |
|------|---------|---------|---------|------|------|------|
| baseline | 0.2432 | 0.1840 | 0.1656 | 39 | 0 | 509ms |
| rag | 0.2432 | 0.1840 | 0.1558 | 45 | 1 | 798ms |
| harness | 0.2432 | 0.1840 | 0.1569 | 47 | 2 | 1416ms |

#### Prompt v2 结果（推荐）

| 模式 | 原始CER | 规则CER | LLM CER | 改善 | 劣化 | 耗时 |
|------|---------|---------|---------|------|------|------|
| baseline | 0.2432 | 0.1840 | 0.1652 | 40 | 0 | 502ms |
| **rag** | 0.2432 | 0.1840 | **0.0852** | **82** | 3 | 1700ms |
| harness | 0.2432 | 0.1840 | 0.1385 | 50 | 3 | 2603ms |

全链路CER从ElevenLabs原始0.2432降至 **RAG + v2 Prompt 的 0.0852**，降幅**65.0%**。

## 测试

### 运行全部测试

```bash
pytest tests/ -v
```

### 运行特定层测试

```bash
# Layer 1
pytest tests/test_preprocessor.py tests/test_itn_rules.py -v

# Layer 2
pytest tests/test_dictionary.py -v

# Layer 3
pytest tests/test_context.py -v

# Layer 4
pytest tests/test_semantic.py -v

# 流水线集成
pytest tests/test_pipeline.py -v

# 端到端评估
pytest tests/test_e2e.py -v -s
```

### 性能基准

```bash
python tests/benchmark.py --samples 100
```

输出 P50/P95/P99 延迟、内存占用、批量吞吐量，报告保存到 `reports/`。

## 数据维护

### 词典更新

```bash
# 1. 从错误反馈生成待审核列表
python scripts/update_lexicon.py --review --output pending_aliases.csv

# 2. 人工审核 CSV（填写 approved 列为 yes/no）

# 3. 导入审核结果
python scripts/update_lexicon.py --import-csv approved_aliases.csv
```

### 配置化纠错规则

纠错映射表已从代码中提取为配置文件，修改规则无需改代码：

| 配置文件 | 用途 | 修改方式 |
|----------|------|----------|
| `data/lexicon/word_confusion.json` | 词语级混淆映射（如"道差"→"道岔"） | 直接编辑JSON，以`_`开头的key为注释 |
| `data/lexicon/phrase_patterns.json` | 短语级纠错模式（如数字归一化、标点修复） | 直接编辑JSON数组 |
| `data/lexicon/aliases.json` | 术语别名映射 | 通过 `update_lexicon.py` 或直接编辑 |

### N-gram 重训练

```bash
# 语料更新后重训练
python scripts/retrain_ngram.py

# 强制替换（跳过困惑度对比）
python scripts/retrain_ngram.py --force
```

### 语料扩增

```bash
# 从新文本中提取候选语料
python scripts/augment_corpus.py --input new_texts.txt --output data/corpus/railway_corpus.txt
```

## 关键设计

### 实体安全校验

Layer 4 的 LLM 输出需经过 `EntityGuard` 校验，确保关键数字实体不被篡改：

- 车次号：`G1023次` → 数字 `1023` 不可变
- 道岔号：`18号道岔` → 数字 `18` 不可变
- 股道号：`3道` → 数字 `3` 不可变
- 公里标：`K134+800` → 不可变
- 限速：`限速60km/h` → 数字 `60` 不可变
- 时间：`X点X分` → 不可变

校验失败时自动回退到 Layer 3 结果。

### 术语工具 (TermTool)

`term_tool.py` 从 railway_terms、aliases、word_confusion、hotwords 四个数据源构建拼音索引，支持：

- **精确拼音匹配**：拼音完全相同 → 相似度1.0
- **模糊拼音匹配**：逐音节比较，相似度≥0.6 → 按置信度分级（高≥0.9 / 中≥0.7 / 低≥0.6）

用于在LLM调用前预检索文本中的可疑词，将拼音相似的铁路术语候选注入Prompt。

### 领域知识检索 (KnowledgeRetriever)

`knowledge_retriever.py` 从三个维度检索与当前文本相关的领域知识：

| 知识类型 | 数据来源 | 用途 |
|----------|----------|------|
| 术语映射 | aliases.json + word_confusion.json | 已验证的标准对应关系 |
| 历史错误模式 | asr_error_pairs.jsonl | 常见ASR误识别模式及频率 |
| 场景规则 | railway_terms.json | 当前文本所属场景的标准术语集 |

### 别名替换策略

- 最长匹配优先：按别名长度降序匹配
- 非重叠替换：已被更长别名覆盖的位置不再匹配
- 全词保护：仅单字别名（<=1字）自动启用全词匹配，避免在长词内部误触
- 正则模式兜底：对标准术语生成容错 pattern，主动替换未收录的别名变体

### Prompt 版本管理

Prompt 文件存储在 `src/prompts/` 目录，支持多版本切换：

| 版本 | 文件 | 特点 | 最佳搭配 |
|------|------|------|----------|
| v1 | `src/prompts/v1/system.txt` | 原则引导式，明确纠错范围和严禁操作 | baseline / rag |
| v2 | `src/prompts/v2/system.txt` | 铁路专家角色，含术语体系、推理铁律、场景分类、Few-shot示例 | **rag（强烈推荐）** |

```bash
# 使用 v1 版本（默认）
export LLM_PROMPT_VERSION=v1

# 使用 v2 版本（RAG模式下效果最优）
export LLM_PROMPT_VERSION=v2
```

```python
import os
os.environ["LLM_PROMPT_VERSION"] = "v2"

from src.pipeline import PostCorrectionPipeline
pipeline = PostCorrectionPipeline()
result = pipeline.run(text, layers=[1, 2, 3], enable_semantic=True, semantic_mode="rag")
```

版本注册表 `registry.json` 记录每个版本的创建时间、描述和评估指标。

## 技术依赖

| 依赖 | 用途 |
|------|------|
| jieba | 中文分词（TermTool预检索 + 错误模式分析） |
| pypinyin | 拼音转换（术语工具拼音索引 + 拼音混淆候选） |
| cn2an | 中文数字转阿拉伯数字 |
| openai | LLM API 调用（OpenAI 兼容接口） |
| rapidfuzz | 快速编辑距离计算（评估指标） |
| pandas | 评估报告数据处理 |
| gradio | Web交互式演示界面 |
| python-dotenv | .env环境变量加载 |
| openpyxl | Excel报告生成（条件格式高亮） |
| pytest | 测试框架 |
