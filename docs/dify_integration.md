# Dify 集成配置文档

本文档说明如何在 Dify 上管理热词、Prompt 和领域知识，并与 ASR 后处理纠错服务同步。

## 1. 集成架构

```
运营人员 --编辑--> Dify 知识库 --同步触发--> ASR 服务 /api/v1/dify-sync/* --拉取--> 本地文件
                                                          |
                                                          ▼
                                                  纠错 API /api/v1/correct
```

关键原则：

- **Dify 仅作为编辑/协作入口**，不进入实时纠错链路。
- **本地文件为权威源**，Dify 同步失败不影响当前服务运行。
- **拉模式同步**：由 ASR 服务主动从 Dify 拉取内容。

## 2. ASR 服务配置

在 `.env` 中启用并配置 Dify：

```ini
DIFY_ENABLED=true
DIFY_BASE_URL=http://your-dify-server:5001
DIFY_API_KEY=dataset-xxxxxxxx
DIFY_HOTWORDS_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DIFY_PROMPTS_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DIFY_ALIASES_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DIFY_KNOWLEDGE_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DIFY_SYNC_INTERVAL_SECONDS=300
```

获取 Dataset API Key：

1. 进入 Dify 知识库页面。
2. 点击左下角 **Service API**。
3. 创建 API Key，格式为 `dataset-xxx`。

## 3. 热词管理

### 3.1 Dify 侧

1. 创建知识库 `asr-hotwords`。
2. 每个文档代表一个分类，例如 `locomotives.txt`、`procedures.txt`。
3. 文档内容每行一个热词，例如：

```text
DF11G型内燃机车
HXD1D型电力机车
SS9型电力机车
```

### 3.2 同步

手动触发：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/hotwords/pull" \
  -H "Content-Type: application/json"
```

返回示例：

```json
{
  "status": "ok",
  "dataset_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "updated": 12,
  "deleted": 0,
  "skipped": 0
}
```

## 4. Prompt 管理

### 4.1 Dify 侧

1. 创建知识库 `asr-prompts`。
2. 每个 Prompt 版本拆成两个文档：
   - `{version}_system.txt`，例如 `v2_system.txt`
   - `{version}_user_template.txt`，例如 `v2_user_template.txt`

### 4.2 同步

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/prompts/pull" \
  -H "Content-Type: application/json"
```

同步后可通过 `/api/v1/prompts` 查看版本，通过 `/api/v1/prompts/{version}/set-default` 切换默认版本。

## 5. 正别名映射知识库

### 5.1 Dify 侧

1. 创建知识库 `asr-aliases`。
2. 上传文档，支持两种格式：

**格式 A：JSON 文件**（推荐批量管理）

文档名：`aliases.json`

内容：

```json
{
  "道差": "道岔",
  "新号机": "信号机",
  "消记": "销记"
}
```

**格式 B：文本文件**

每行一个映射，支持 `->` 或 `|` 分隔：

```text
# 道岔相关
道差 -> 道岔
到岔 -> 道岔

# 信号相关
新号机 | 信号机
信后机 | 信号机
```

### 5.2 同步

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/aliases/pull" \
  -H "Content-Type: application/json"
```

返回示例：

```json
{
  "status": "ok",
  "dataset_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "count": 1384,
  "path": "data/lexicon/aliases.json"
}
```

同步后会：

- 备份原 `aliases.json` 到 `data/lexicon/backups/aliases.json.bak`
- 覆盖本地 `data/lexicon/aliases.json`
- 运行时热重载 `DictionaryCorrector`、`PhoneticCandidateGenerator` 以及 Pipeline 中的 `RAGRefiner` / `HarnessRefiner` 术语索引

无需重启服务即可生效。

## 6. 领域知识/场景知识库（DIFY_KNOWLEDGE_DATASET_ID）

`DIFY_KNOWLEDGE_DATASET_ID` 设计用于存放**无法简单用“正别名映射”表达**的复杂领域知识，例如：

- 历史 ASR 错误对（`asr_error_pairs.jsonl`）
- 铁路调度场景规则说明
- 特定作业流程的术语消歧说明

注意：当前 `/api/v1/dify-sync/knowledge/pull` 仅返回文档预览，**尚未实现自动解析持久化**。若需要把场景规则也纳入 Dify 管理，建议先按“正别名映射知识库”的方式管理，或后续扩展 `knowledge/pull` 实现。

## 7. Dify Workflow 配置（推荐）

### 7.1 创建 `asr-config-sync` Workflow

1. **Start Node**：输入 `sync_type`（hotwords/prompts/aliases/knowledge）。
2. **HTTP Request Node**：
   - Method：`POST`
   - URL：`http://<asr-service>/api/v1/dify-sync/{{#start.sync_type#}}/pull`
   - Headers：`Content-Type: application/json`
3. **End Node**：返回 HTTP 响应结果。

运营人员编辑完热词或 Prompt 后，点击运行该 Workflow 即可触发同步。

### 7.2 可选：测试 Workflow

创建 `asr-correction-test`：

1. **Start Node**：输入原始 ASR 文本。
2. **HTTP Request Node**：
   - Method：`POST`
   - URL：`http://<asr-service>/api/v1/correct`
   - Body：`{"text": "{{#start.text#}}", "layers": [1, 2, 3]}`
3. **End Node**：展示纠错结果。

## 8. 故障排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| 同步返回 403 | `DIFY_ENABLED=false` | 检查 `.env` |
| 同步返回 400 | API Key 或 Dataset ID 错误 | 检查 Dify 控制台 |
| 同步成功但热词未生效 | 未调用热词重载 | 调用 `/api/v1/hotwords/reload`，同步接口已自动重载 |
| Prompt 同步后未生效 | 未切换默认版本 | 调用 `/api/v1/prompts/{version}/set-default` |
| 别名同步后未生效 | 流水线缓存了旧结果 | 首次新请求即生效；如验证可调用 `/api/v1/correct` 测试 |

## 9. 注意事项

- Dify 知识库中的文档更新后，可能需要等待索引完成才能被 Segment API 读取。
- 建议先在测试环境验证 Dify Workflow，再接入生产。
- 若 Dify 不可用，服务会自动使用本地最新配置继续运行。
