# API 接口说明

服务启动后，完整的 OpenAPI 文档可通过以下地址访问：

- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`
- OpenAPI JSON：`http://localhost:8000/openapi.json`

主要接口分组：

- **health**：健康检查 `/health`、就绪检查 `/ready`
- **correction**：单条纠错 `/api/v1/correct`、批量纠错 `/api/v1/correct/batch`
- **hotwords**：热词 CRUD `/api/v1/hotwords/*`
- **prompts**：Prompt 版本管理 `/api/v1/prompts/*`
- **dify-sync**：Dify 同步 `/api/v1/dify-sync/*`
- **config**：服务信息 `/api/v1/info`、配置快照 `/api/v1/config`、指标 `/api/v1/metrics`

所有需要鉴权的接口默认不校验（当未配置 `API_KEY` 时）。若配置了 `API_KEY`，请在请求头中携带：

```
X-API-Key: your-api-key
```
