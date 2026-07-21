"""API 请求/响应模型."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """单条纠错请求."""

    text: str = Field(..., description="ASR 原始输出文本", min_length=1, max_length=4096)
    layers: Optional[List[int]] = Field(default=None, description="启用的层号列表，如 [1,2,3]")
    enable_semantic: bool = Field(default=True, description="是否启用 Layer 4 语义精修")
    semantic_mode: str = Field(default="rag", description="语义精修模式：baseline/rag/harness")
    hotwords_override: Optional[List[str]] = Field(default=None, description="本次请求临时热词列表")


class CorrectionDetail(BaseModel):
    """单步修改详情."""

    layer: str
    changes: List[dict]


class CorrectionResponse(BaseModel):
    """单条纠错响应."""

    original: str
    corrected: str
    layers_applied: List[str]
    layer_outputs: Dict[str, str]
    details: List[CorrectionDetail]
    latency_ms: float


class BatchCorrectionRequest(BaseModel):
    """批量纠错请求."""

    items: List[CorrectionRequest] = Field(..., max_length=50, description="最多 50 条")


class BatchCorrectionResponse(BaseModel):
    """批量纠错响应."""

    results: List[CorrectionResponse]
    total_latency_ms: float


class HotwordItem(BaseModel):
    """热词项."""

    id: str
    word: str
    category: Optional[str] = None
    enabled: bool = True


class HotwordCreateRequest(BaseModel):
    """创建热词请求."""

    word: str = Field(..., min_length=1, max_length=256)
    category: Optional[str] = None
    enabled: bool = True


class HotwordUpdateRequest(BaseModel):
    """更新热词请求."""

    word: Optional[str] = Field(default=None, min_length=1, max_length=256)
    category: Optional[str] = None
    enabled: Optional[bool] = None


class HotwordsResponse(BaseModel):
    """热词列表响应."""

    total: int
    items: List[HotwordItem]


class PromptVersion(BaseModel):
    """Prompt 版本信息."""

    version: str
    description: str
    created_at: str
    is_default: bool


class PromptContent(BaseModel):
    """Prompt 内容."""

    version: str
    system: str
    user_template: str


class PromptUpdateRequest(BaseModel):
    """更新 Prompt 请求."""

    system: Optional[str] = None
    user_template: Optional[str] = None
    description: Optional[str] = None


class ServiceInfo(BaseModel):
    """服务信息."""

    name: str
    version: str
    llm_model: str
    prompt_version: str
    capabilities: List[str]


class HealthResponse(BaseModel):
    """健康检查响应."""

    status: str


class ReadyResponse(BaseModel):
    """就绪检查响应."""

    status: str
    checks: Dict[str, bool]
