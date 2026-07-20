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

## 5. Dify Workflow 配置（推荐）

### 5.1 创建 `asr-config-sync` Workflow

1. **Start Node**：输入 `sync_type`（hotwords/prompts/knowledge）。
2. **HTTP Request Node**：
   - Method：`POST`
   - URL：`http://<asr-service>/api/v1/dify-sync/{{#start.sync_type#}}/pull`
   - Headers：`Content-Type: application/json`
3. **End Node**：返回 HTTP 响应结果。

运营人员编辑完热词或 Prompt 后，点击运行该 Workflow 即可触发同步。

### 5.2 可选：测试 Workflow

创建 `asr-correction-test`：

1. **Start Node**：输入原始 ASR 文本。
2. **HTTP Request Node**：
   - Method：`POST`
   - URL：`http://<asr-service>/api/v1/correct`
   - Body：`{"text": "{{#start.text#}}", "layers": [1, 2, 3]}`
3. **End Node**：展示纠错结果。

## 6. 故障排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| 同步返回 403 | `DIFY_ENABLED=false` | 检查 `.env` |
| 同步返回 400 | API Key 或 Dataset ID 错误 | 检查 Dify 控制台 |
| 同步成功但热词未生效 | 未调用热词重载 | 调用 `/api/v1/hotwords/reload`，同步接口已自动重载 |
| Prompt 同步后未生效 | 未切换默认版本 | 调用 `/api/v1/prompts/{version}/set-default` |

## 7. 注意事项

- Dify 知识库中的文档更新后，可能需要等待索引完成才能被 Segment API 读取。
- 建议先在测试环境验证 Dify Workflow，再接入生产。
- 若 Dify 不可用，服务会自动使用本地最新配置继续运行。
