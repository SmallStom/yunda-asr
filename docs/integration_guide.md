# ASR 纠错服务 - 系统集成文档

> 本文档说明外部系统如何与 ASR 纠错服务集成，包括知识库维护、语音纠错接口调用、版本切换三个部分。

## 基础信息

| 项目 | 说明 |
|---|---|
| 服务地址 | `http://<服务器IP>:8023` |
| API 前缀 | `/api/v1` |
| 鉴权 | 默认无鉴权，直接调用 |
| 请求格式 | JSON（纠错接口）/ multipart-form（语音接口） |

## 一、通过 Dify 维护知识库

### 1.1 整体架构

```
Dify 知识库（运营人员编辑）
      │
      ▼  HTTP 调用同步接口
ASR 纠错服务（拉取并热重载）
      │
      ▼  立即生效
纠错流水线（使用新配置）
```

### 1.2 创建 Dify 知识库

在 Dify 中创建 **3 个知识库**：

| 知识库 | 用途 | 文档格式 |
|---|---|---|
| `asr-hotwords` | 热词 | TXT（每行一个词） |
| `asr-prompts` | LLM 提示词 | TXT（system.txt 内容） |
| `asr-aliases` | 正别名映射 | JSON 或 TXT |

### 1.3 文档命名规范

**一个文档 = 一个场景版本**。文档名（不含扩展名）即为版本名。

例如热词知识库中上传 3 个文档：

| 文档名 | 内容 | 版本名 |
|---|---|---|
| `调度.txt` | 调度场景的热词，每行一个 | `调度` |
| `检修.txt` | 检修场景的热词 | `检修` |
| `行车.txt` | 行车场景的热词 | `行车` |

> 同步时通过 `version` 参数指定要拉取哪个文档，系统精确匹配文档名。

### 1.4 各知识库文档格式

#### 热词文档（`调度.txt`）

```
道岔
信号机
轨道电路
接触网
闭塞分区
```

每行一个热词，空行忽略。

#### 提示词文档（`调度_system.txt`）

直接填写 system prompt 内容，例如：

```
你是铁路调度领域的ASR文本纠错专家。请根据以下规则修正文本...
```

文档名规范：`{版本名}_system.txt`，如 `调度_system.txt`。

#### 别名文档（`调度.json`）

JSON 格式：

```json
{
  "道差": "道岔",
  "新号机": "信号机",
  "消记": "销记"
}
```

或文本格式（`调度.txt`）：

```
道差 -> 道岔
新号机 | 信号机
消记 -> 销记
```

### 1.5 同步接口

同步接口统一为 POST 请求，JSON Body 传参：

```json
{
  "dataset_id": "Dify知识库ID",
  "version": "版本名"
}
```

| 接口 | URL | 说明 |
|---|---|---|
| 同步热词 | `POST /api/v1/dify-sync/hotwords/pull` | 从 Dify 拉取热词 |
| 同步提示词 | `POST /api/v1/dify-sync/prompts/pull` | 从 Dify 拉取 Prompt |
| 同步别名 | `POST /api/v1/dify-sync/aliases/pull` | 从 Dify 拉取别名 |

#### 请求示例

```bash
curl -X POST "http://192.168.1.119:8023/api/v1/dify-sync/hotwords/pull" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "74d6ae75-5a28-443a-b8ab-56ce9cdb011b", "version": "调度"}'
```

#### 响应示例

```json
{
  "status": "ok",
  "dataset_id": "74d6ae75-5a28-443a-b8ab-56ce9cdb011b",
  "version": "调度",
  "count": 874,
  "deleted": 3,
  "skipped": 0
}
```

| 字段 | 说明 |
|---|---|
| `count` | 本次同步写入的条目数 |
| `deleted` | 上一次同步操作写入的条目数（全量替换语义） |
| `skipped` | 无法解析的条目数（仅热词有） |

三个接口响应格式统一，`count` 始终表示同步数量。

### 1.6 激活版本

同步只生成本地版本文件，**不影响线上**。需要显式激活才生效。

| 资源 | 激活接口 |
|---|---|
| 热词 | `POST /api/v1/hotwords/switch-version?version=调度` |
| 别名 | `POST /api/v1/aliases/switch-version?version=调度` |
| 提示词 | `POST /api/v1/prompts/调度/set-default` |

```bash
# 激活调度版本的热词
curl -X POST "http://192.168.1.119:8023/api/v1/hotwords/switch-version?version=调度"

# 激活调度版本的别名
curl -X POST "http://192.168.1.119:8023/api/v1/aliases/switch-version?version=调度"

# 激活调度版本的提示词
curl -X POST "http://192.168.1.119:8023/api/v1/prompts/调度/set-default"
```

激活后立即生效，无需重启服务。切换到其他版本同理。

### 1.7 Dify Workflow 配置

在 Dify 中创建 Workflow，串联同步接口：

**Start 节点**：定义变量 `dataset_id`（string）、`version`（string）

**HTTP Request 节点**（可串联 3 个）：

| 配置项 | 值 |
|---|---|
| Method | `POST` |
| URL | `http://<服务器IP>:8023/api/v1/dify-sync/hotwords/pull` |
| Headers | `Content-Type: application/json` |
| Body 类型 | `JSON` |
| Body 内容 | `{"dataset_id": "{{#start.dataset_id#}}", "version": "{{#start.version#}}"}` |

运行时输入 `version=调度`，一键同步。

### 1.8 完整操作流程

```
1. 在 Dify 知识库编辑文档（如修改 调度.txt）
2. 调用同步接口（或运行 Dify Workflow）-> 生成本地版本文件
3. 测试验证纠错效果
4. 调用激活接口 -> 线上立即生效
5. 如需回滚 -> 激活其他版本
```

## 二、调用语音转文本纠错接口

### 2.1 接口说明

上传音频文件，自动完成：音频 → [可选降噪] → ASR转文本 → 四级纠错 → 输出纠错文本。

| 项目 | 说明 |
|---|---|
| URL | `POST /api/v1/transcribe-and-correct` |
| 请求格式 | `multipart/form-data` |
| 响应格式 | `JSON` |

### 2.2 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `file` | 文件 | 是 | - | 音频文件（wav/mp3/flac/m4a） |
| `enable_denoise` | bool | 否 | `false` | 是否开启 DeepFilterNet 降噪 |
| `layers` | string | 否 | null | 启用的层号，逗号分隔，如 `1,2,3`。不传则走全部层 |
| `enable_semantic` | bool | 否 | `true` | 是否启用 Layer 4 语义精修（LLM） |
| `semantic_mode` | string | 否 | `rag` | 语义精修模式：`baseline`/`rag`/`harness` |

> `semantic_mode` 只有显式传 `baseline` 才走 baseline，其他所有值（含无效值）都优先走 `rag`。

### 2.3 调用示例

#### curl

```bash
curl -X POST "http://192.168.1.119:8023/api/v1/transcribe-and-correct" \
  -F "file=@audio.wav" \
  -F "enable_denoise=false" \
  -F "enable_semantic=true" \
  -F "semantic_mode=rag"
```

#### Python

```python
import requests

url = "http://192.168.1.119:8023/api/v1/transcribe-and-correct"
files = {"file": open("audio.wav", "rb")}
data = {
    "enable_denoise": "false",
    "enable_semantic": "true",
    "semantic_mode": "rag",
}

resp = requests.post(url, files=files, data=data)
result = resp.json()
print(result["corrected"])
```

#### Java

```java
OkHttpClient client = new OkHttpClient();

RequestBody body = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "audio.wav",
        RequestBody.create(new File("audio.wav"), MediaType.parse("audio/wav")))
    .addFormDataPart("enable_denoise", "false")
    .addFormDataPart("enable_semantic", "true")
    .addFormDataPart("semantic_mode", "rag")
    .build();

Request request = new Request.Builder()
    .url("http://192.168.1.119:8023/api/v1/transcribe-and-correct")
    .post(body)
    .build();

Response response = client.newCall(request).execute();
System.out.println(response.body().string());
```

### 2.4 响应格式

```json
{
  "status": "ok",
  "original_audio": "audio.wav",
  "asr_text": "报告值班员，十三号道岔五表示。",
  "corrected": "报告值班员，13号道岔五表示。",
  "layers_applied": ["preprocessor", "dictionary", "context", "semantic"],
  "layer_outputs": {
    "layer1": "报告值班员，13号道岔五表示。",
    "layer2": "报告值班员，13号道岔五表示。",
    "layer3": "报告值班员，13号道岔五表示。",
    "layer4": "报告值班员，13号道岔五表示。"
  },
  "details": [
    {
      "layer": "preprocessor",
      "changes": [
        {
          "layer": "itn",
          "rule": "numbered_term",
          "before": "十三号",
          "after": "13号"
        }
      ]
    }
  ],
  "steps": {
    "asr_latency_ms": 130,
    "correct_latency_ms": 1586
  },
  "total_latency_ms": 1716
}
```

| 字段 | 说明 |
|---|---|
| `asr_text` | ASR 识别出的原始文本 |
| `corrected` | 纠错后的最终文本 |
| `layers_applied` | 实际执行的层列表 |
| `layer_outputs` | 各层处理后的文本 |
| `details` | 各层的修改详情 |
| `steps` | 各步骤耗时 |
| `total_latency_ms` | 总耗时 |

### 2.5 纠错层级说明

| 层 | 名称 | 说明 |
|---|---|---|
| Layer 1 | 预处理 | 标点补全、数字格式化（"十三号"→"13号"） |
| Layer 2 | 词典纠错 | 基于别名的精确替换（"道差"→"道岔"） |
| Layer 3 | 上下文纠错 | 拼音相似度匹配 + 上下文消歧 |
| Layer 4 | 语义精修 | LLM 基于 RAG 检索热词/别名做最终修正 |

### 2.6 纯文本纠错接口

如果已经有 ASR 文本，只需纠错，可直接调用：

```bash
curl -X POST "http://192.168.1.119:8023/api/v1/correct" \
  -H "Content-Type: application/json" \
  -d '{"text": "报告值班员，十三号道岔五表示。"}'
```

## 三、版本切换与重置

### 3.1 查看可用版本

```bash
# 查看热词版本
curl "http://192.168.1.119:8023/api/v1/hotwords/versions"

# 查看别名版本
curl "http://192.168.1.119:8023/api/v1/aliases/versions"

# 查看提示词版本
curl "http://192.168.1.119:8023/api/v1/prompts"
```

### 3.2 切换版本

切换版本 = 激活指定版本，立即生效：

```bash
# 切换热词
curl -X POST "http://192.168.1.119:8023/api/v1/hotwords/switch-version?version=检修"

# 切换别名
curl -X POST "http://192.168.1.119:8023/api/v1/aliases/switch-version?version=检修"

# 切换提示词
curl -X POST "http://192.168.1.119:8023/api/v1/prompts/检修/set-default"
```

### 3.3 重置为默认

重置 = 重新从 Dify 同步并激活。操作流程与第一部分相同：

```bash
# 1. 重新同步默认版本
curl -X POST "http://192.168.1.119:8023/api/v1/dify-sync/hotwords/pull" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "热词知识库ID", "version": "调度"}'

# 2. 激活
curl -X POST "http://192.168.1.119:8023/api/v1/hotwords/switch-version?version=调度"
```

> 重置和同步是同一个流程，没有独立的"重置"接口。

## 四、附录

### 4.1 环境配置

ASR 服务通过 `.env` 文件配置，关键项：

```ini
# 服务端口（宿主机端口，容器内固定8000）
HOST_PORT=8023

# ASR 服务（OpenAI 兼容接口）
QWEN3_ASR_BASE_URL=http://192.168.1.119:8014
QWEN3_ASR_MODEL=/models/Qwen3-ASR-1.7B
QWEN3_ASR_API_KEY=dummy-key-for-local

# LLM 服务（Layer 4 语义精修）
LLM_BASE_URL=http://192.168.1.119:8012/v1
LLM_MODEL=Qwen3.6-27B
LLM_API_KEY=dummy-key-for-local

# Dify 集成
DIFY_ENABLED=true
DIFY_BASE_URL=http://192.168.1.120:80
DIFY_API_KEY=app-xxx
```

### 4.2 错误码

| HTTP 状态码 | 说明 |
|---|---|
| 200 | 成功 |
| 400 | Dify 客户端错误（知识库ID无效、连接失败等） |
| 403 | Dify 集成未启用 |
| 422 | 请求参数校验失败 |
| 500 | 服务内部错误 |

### 4.3 接口速查表

| 功能 | 方法 | URL |
|---|---|---|
| 语音纠错 | POST | `/api/v1/transcribe-and-correct` |
| 文本纠错 | POST | `/api/v1/correct` |
| 同步热词 | POST | `/api/v1/dify-sync/hotwords/pull` |
| 同步提示词 | POST | `/api/v1/dify-sync/prompts/pull` |
| 同步别名 | POST | `/api/v1/dify-sync/aliases/pull` |
| 切换热词版本 | POST | `/api/v1/hotwords/switch-version?version=xxx` |
| 切换别名版本 | POST | `/api/v1/aliases/switch-version?version=xxx` |
| 切换提示词版本 | POST | `/api/v1/prompts/{version}/set-default` |
| 查看热词版本 | GET | `/api/v1/hotwords/versions` |
| 查看别名版本 | GET | `/api/v1/aliases/versions` |
| 查看提示词版本 | GET | `/api/v1/prompts` |
| 健康检查 | GET | `/health` |
