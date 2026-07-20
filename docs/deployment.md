# ASR 后处理纠错服务部署文档

## 1. 环境要求

- Python >= 3.9（推荐 3.11）
- 可选：Docker / Docker Compose
- 内网可访问的 LLM 服务（OpenAI 兼容接口）
- 可选：Dify（用于 Prompt/热词/知识库管理）

## 2. 本地直接部署

### 2.1 安装依赖

```bash
cd asr
pip install -e .
```

如果需要音频处理能力：

```bash
pip install -e ".[audio]"
```

### 2.2 配置环境变量

复制 `.env.example` 为 `.env`，并根据实际情况修改：

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_BASE_URL` | LLM OpenAI 兼容接口地址 | `http://192.168.1.119:8012/v1` |
| `LLM_MODEL` | LLM 模型名 | `Qwen3.6-27B` |
| `LLM_PROMPT_VERSION` | 默认 Prompt 版本 | `v2` |
| `API_HOST` | API 服务监听地址 | `0.0.0.0` |
| `API_PORT` | API 服务端口 | `8000` |
| `API_KEY` | 可选的 API Key 鉴权 | 空（不鉴权） |

### 2.3 启动服务

```bash
python scripts/start_api.py
```

服务启动后访问：

- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

### 2.4 测试纠错接口

```bash
curl -X POST "http://localhost:8000/api/v1/correct" \
  -H "Content-Type: application/json" \
  -d '{"text": "十八号道差开通反位，信号好了", "layers": [1, 2, 3]}'
```

## 3. Docker 部署

### 3.1 构建并启动

```bash
docker-compose up -d --build
```

### 3.2 查看日志

```bash
docker-compose logs -f asr-correction
```

### 3.3 停止服务

```bash
docker-compose down
```

## 4. 热词推送到上游 ASR

当前 `/api/v1/hotwords/push-to-asr` 仅生成 ASR 可用的热词格式，具体推送需根据上游 ASR 协议实现。

示例返回：

```json
{
  "status": "prepared",
  "count": 874,
  "payload": {
    "hotwords": ["30kPa单阀缓解量", "6502电气集中联锁"],
    "categories": {
      "default": ["30kPa单阀缓解量"],
      "locomotives": ["DF11G型内燃机车"]
    }
  }
}
```

## 5. 生产建议

- 建议设置 `API_KEY` 进行基础鉴权。
- 建议配置反向代理（Nginx）并启用 HTTPS。
- 建议通过日志收集器采集 JSON 日志。
- 建议定期备份 `data/lexicon/hotwords.json` 和 `src/prompts/`。
