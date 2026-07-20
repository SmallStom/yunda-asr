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

1. 创建知识库 `asr-hotwords`（**只需一个知识库**）。
2. 在知识库内通过**文档名区分版本/场景**，命名规则：
   - `{版本名}.txt` - 一个文档即一个版本，内容为该版本全部热词
   - `{版本名}_{分类}.txt` - 一个版本拆成多个分类文档

   例如同时管理"调度"、"检修"、"行车"三套热词：

   | 文档名 | 说明 |
   |---|---|
   | `调度.txt` | 调度场景全部热词 |
   | `检修.txt` | 检修场景全部热词 |
   | `行车.txt` | 行车场景全部热词 |
   | `调度_机车.txt` | 调度场景-机车分类（可选拆分） |
   | `调度_信号.txt` | 调度场景-信号分类（可选拆分） |

3. 文档内容每行一个热词：

```text
DF11G型内燃机车
HXD1D型电力机车
SS9型电力机车
```

### 3.2 同步

**同步指定版本**（只拉取该版本的文档，保存为版本文件，不影响线上）：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/hotwords/pull?version=调度"
```

**同步全部**（不指定 version，拉取知识库内所有文档并合并为活跃文件）：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/hotwords/pull"
```

返回示例：

```json
{
  "status": "ok",
  "dataset_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "version": "调度",
  "updated": 12,
  "deleted": 0,
  "skipped": 0
}
```

## 4. Prompt 管理

### 4.1 Dify 侧

1. 创建知识库 `asr-prompts`（**只需一个知识库**）。
2. 每个 Prompt 版本拆成两个文档，**文档名即为版本名**：
   - `{版本名}_system.txt`，例如 `调度_system.txt`、`检修_system.txt`
   - `{版本名}_user_template.txt`，例如 `调度_user_template.txt`、`检修_user_template.txt`

### 4.2 同步

**同步指定版本**：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/prompts/pull?version=调度"
```

**同步全部版本**：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/prompts/pull"
```

同步后可通过 `/api/v1/prompts` 查看版本，通过 `/api/v1/prompts/{version}/set-default` 切换默认版本。

## 5. 正别名映射知识库

### 5.1 Dify 侧

1. 创建知识库 `asr-aliases`（**只需一个知识库**）。
2. 通过**文档名区分版本**，命名规则：
   - `{版本名}.json` - JSON 格式，内容为完整的别名字典
   - `{版本名}.txt` - 文本格式，每行一个映射
   - `{版本名}_{子集}.txt` - 版本内拆分多个文档（可选）

   例如：

   | 文档名 | 格式 | 说明 |
   |---|---|---|
   | `调度.json` | JSON | 调度场景别名映射 |
   | `检修.json` | JSON | 检修场景别名映射 |
   | `行车.txt` | 文本 | 行车场景别名映射 |

**JSON 格式**内容示例：

```json
{
  "道差": "道岔",
  "新号机": "信号机",
  "消记": "销记"
}
```

**文本格式**每行一个映射，支持 `->` 或 `|` 分隔：

```text
# 道岔相关
道差 -> 道岔
到岔 -> 道岔

# 信号相关
新号机 | 信号机
信后机 | 信号机
```

### 5.2 同步

**同步指定版本**（只拉取该版本的文档，保存为版本文件，不影响线上）：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/aliases/pull?version=调度"
```

**同步全部**（不指定 version，拉取知识库内所有文档并合并为活跃文件）：

```bash
curl -X POST "http://localhost:8000/api/v1/dify-sync/aliases/pull"
```

返回示例：

```json
{
  "status": "ok",
  "dataset_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "version": "调度",
  "count": 1384,
  "path": "data/lexicon/aliases_调度.json"
}
```

指定 `version` 时保存为版本文件（如 `aliases_调度.json`），不覆盖活跃文件；不指定 `version` 时覆盖活跃 `aliases.json` 并热重载。

无需重启服务即可生效。

## 6. 领域知识/场景知识库（DIFY_KNOWLEDGE_DATASET_ID）

`DIFY_KNOWLEDGE_DATASET_ID` 设计用于存放**无法简单用“正别名映射”表达**的复杂领域知识，例如：

- 历史 ASR 错误对（`asr_error_pairs.jsonl`）
- 铁路调度场景规则说明
- 特定作业流程的术语消歧说明

注意：当前 `/api/v1/dify-sync/knowledge/pull` 仅返回文档预览，**尚未实现自动解析持久化**。若需要把场景规则也纳入 Dify 管理，建议先按“正别名映射知识库”的方式管理，或后续扩展 `knowledge/pull` 实现。

## 7. Dify Workflow 配置（推荐）

下面给出两种常用 Workflow：

- `asr-config-sync`：运营人员在 Dify 编辑完热词/Prompt/别名后，一键同步到 ASR 服务。
- `asr-correction-test`：在 Dify 里快速测试纠错效果。

### 7.1 前置检查

1. ASR 服务已启动，且 `.env` 中 `DIFY_ENABLED=true`。
2. API 鉴权（可选）：`.env` 中 `API_KEY` **默认未配置（注释状态）**，服务开放无需鉴权。如果你出于安全考虑配置了 `API_KEY=xxx`，则 Workflow 的 HTTP 请求必须带请求头 `x-api-key: xxx`。不配置则无需此头。
3. Dify 能访问到 ASR 服务的地址（内网 IP / 域名 + 端口）。

### 7.2 创建 `asr-config-sync` Workflow

**步骤 1：新建 Workflow**

进入 Dify → 工作室 → 创建空白应用 → 选择 **Workflow（工作流）** → 命名 `asr-config-sync`。

**步骤 2：配置 Start Node**

点击 **Start** 节点，添加一个输入变量：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `sync_type` | `select` / `string` | 是 | 可选值：`hotwords`、`prompts`、`aliases`、`knowledge` |

> 建议使用 `select` 类型，把四个选项写死，避免运营人员输错。

**步骤 3：添加 HTTP Request Node**

点击 **+** 添加节点 → 选择 **HTTP 请求**。

配置如下：

| 配置项 | 值 |
|---|---|
| 方法 | `POST` |
| URL | `http://<asr-service-host>:8000/api/v1/dify-sync/{{#start.sync_type#}}/pull` |
| Headers | `Content-Type: application/json` |
| Headers（若启用了 API_KEY） | `x-api-key: <your-internal-api-key>` |
| Body | 留空或 `{}` |
| 超时 | 建议 30~60 秒（别名/热词较多时需要时间） |

说明：

- `{{#start.sync_type#}}` 是 Dify 的变量语法，会替换成 Start 节点输入的 `sync_type`。
- `<asr-service-host>` 填写 ASR 服务实际地址，例如 `192.168.1.100` 或 `asr.yunda.local`。
- 如果 Dify 与 ASR 在同一台 Docker 宿主机，且 ASR 映射了宿主机端口，通常用 `host.docker.internal:8000` 或宿主机内网 IP。

**步骤 4：配置 End Node**

添加 **End** 节点，输出字段可以绑定 HTTP 请求的响应：

| 输出字段 | 值 |
|---|---|
| `status` | `{{#http.status_code#}}` |
| `result` | `{{#http.response#}}` |

**步骤 5：保存并发布**

点击右上角 **发布**。

**步骤 6：运行同步**

运营人员在 Dify 知识库编辑完内容后：

1. 进入 `asr-config-sync` Workflow。
2. 在 **运行** 页面选择 `sync_type`（如 `aliases`）。
3. 点击 **运行**。
4. 观察返回结果中的 `status` 是否为 `ok`，`count` 是否符合预期。

### 7.3 创建独立的同步 Workflow（可选）

如果担心运营人员选错 `sync_type`，可以拆成 3 个独立 Workflow：

| Workflow 名称 | URL |
|---|---|
| `asr-sync-hotwords` | `POST http://<asr-service>:8000/api/v1/dify-sync/hotwords/pull` |
| `asr-sync-prompts` | `POST http://<asr-service>:8000/api/v1/dify-sync/prompts/pull` |
| `asr-sync-aliases` | `POST http://<asr-service>:8000/api/v1/dify-sync/aliases/pull` |

每个 Workflow 的 Start 节点不需要输入变量，直接固定 URL 即可。

### 7.4 可选：创建 `asr-correction-test` 测试 Workflow

用于在 Dify 里快速验证某句 ASR 文本的纠错效果。

**Start Node**：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | 是 | ASR 原始文本 |
| `semantic_mode` | select | 否 | 可选：`baseline`、`rag`、`harness`，默认 `baseline` |

**HTTP Request Node**：

| 配置项 | 值 |
|---|---|
| 方法 | `POST` |
| URL | `http://<asr-service>:8000/api/v1/correct` |
| Headers | `Content-Type: application/json` |
| Headers（若启用了 API_KEY） | `x-api-key: <your-internal-api-key>` |
| Body | `{"text": "{{#start.text#}}", "layers": [1, 2, 3], "enable_semantic": true, "semantic_mode": "{{#start.semantic_mode#}}"}` |

**End Node**：

| 输出字段 | 值 |
|---|---|
| `corrected` | `{{#http.response.corrected#}}` |
| `layers_applied` | `{{#http.response.layers_applied#}}` |
| `full_response` | `{{#http.response#}}` |

### 7.5 进一步自动化（可选）

Dify Workflow 目前主要依赖手动点击运行。如果希望定时同步，可以考虑：

- 在 ASR 服务内部开启定时同步（基于 `DIFY_SYNC_INTERVAL_SECONDS` 的轮询任务）。
- 使用外部定时任务（如 cron、K8s CronJob）调用 Dify Workflow 的 API 或 ASR 同步接口。

当前项目尚未实现 ASR 服务内部的定时轮询，如有需要可后续扩展。

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

## 10. 版本管理（热词 / 别名 / Prompt）

**核心设计：一共 3 个知识库（热词/提示词/正别名），通过文档名区分版本。**

版本名可以是任意字符串，例如 `调度`、`检修`、`行车`，或 `v1`、`v2`。

### 10.1 文档命名规范总览

| 知识库 | 文档名格式 | 示例 |
|---|---|---|
| `asr-hotwords` | `{版本}.txt` 或 `{版本}_{分类}.txt` | `调度.txt`、`调度_机车.txt` |
| `asr-prompts` | `{版本}_system.txt` + `{版本}_user_template.txt` | `调度_system.txt`、`调度_user_template.txt` |
| `asr-aliases` | `{版本}.json` 或 `{版本}.txt` | `调度.json`、`检修.txt` |

### 10.2 同步指定版本

`?version=` 参数控制两件事：
1. **从 Dify 拉取哪些文档**：只拉取文档名匹配该版本的文档
2. **本地保存为版本文件**：如 `hotwords_调度.json`，不覆盖活跃文件

```bash
# 同步"调度"版本的热词
curl -X POST "http://localhost:8000/api/v1/dify-sync/hotwords/pull?version=调度"

# 同步"检修"版本的别名
curl -X POST "http://localhost:8000/api/v1/dify-sync/aliases/pull?version=检修"

# 同步"行车"版本的 Prompt
curl -X POST "http://localhost:8000/api/v1/dify-sync/prompts/pull?version=行车"
```

### 10.3 列出与切换版本

**热词**：

```bash
# 列出所有本地版本
curl "http://localhost:8000/api/v1/hotwords/versions"

# 切换激活版本（热重载，无需重启）
curl -X POST "http://localhost:8000/api/v1/hotwords/switch-version?version=调度"
```

**别名**：

```bash
curl "http://localhost:8000/api/v1/aliases/versions"
curl -X POST "http://localhost:8000/api/v1/aliases/switch-version?version=调度"
```

**Prompt**：

```bash
# 查看所有版本
curl "http://localhost:8000/api/v1/prompts"

# 设置默认版本
curl -X POST "http://localhost:8000/api/v1/prompts/调度/set-default"
```

### 10.4 启动时自动激活指定版本

在 `.env` 中配置：

```ini
HOTWORDS_VERSION=调度
ALIASES_VERSION=调度
LLM_PROMPT_VERSION=调度
```

### 10.5 典型版本管理流程

以"调度"场景为例：

1. **在 Dify 知识库中编辑**：上传/修改 `调度.txt`（热词）、`调度.json`（别名）、`调度_system.txt` + `调度_user_template.txt`（Prompt）
2. **同步到本地版本文件**：
   ```bash
   curl -X POST "http://localhost:8000/api/v1/dify-sync/hotwords/pull?version=调度"
   curl -X POST "http://localhost:8000/api/v1/dify-sync/aliases/pull?version=调度"
   curl -X POST "http://localhost:8000/api/v1/dify-sync/prompts/pull?version=调度"
   ```
3. **验证效果**：
   ```bash
   curl -X POST "http://localhost:8000/api/v1/correct" -d '{"text":"十八号道差开通反位"}'
   ```
4. **激活"调度"版本**：
   ```bash
   curl -X POST "http://localhost:8000/api/v1/hotwords/switch-version?version=调度"
   curl -X POST "http://localhost:8000/api/v1/aliases/switch-version?version=调度"
   curl -X POST "http://localhost:8000/api/v1/prompts/调度/set-default"
   ```
5. **如需回滚**：把 `version=` 改成其他版本名重新激活即可。

### 10.6 Dify Workflow 配置（以"调度"为例）

为每个场景创建一个 Workflow，固定 `version` 参数：

**Workflow: `asr-sync-调度`**

- Start Node：无输入变量
- HTTP Request Node（可串联 3 个，或用条件分支）：
  - `POST http://<asr-host>:8000/api/v1/dify-sync/hotwords/pull?version=调度`
  - `POST http://<asr-host>:8000/api/v1/dify-sync/aliases/pull?version=调度`
  - `POST http://<asr-host>:8000/api/v1/dify-sync/prompts/pull?version=调度`
- End Node：返回结果

运营人员编辑完"调度"文档后，运行该 Workflow 即可同步全部三类配置。
