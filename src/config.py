"""全局配置管理.

使用 pydantic-settings 集中管理环境变量，支持 .env 文件加载。
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """应用配置."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API 服务配置
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")

    # LLM 配置
    llm_base_url: str = Field(default="http://192.168.1.119:8012/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="Qwen3.6-27B", alias="LLM_MODEL")
    llm_api_key: str = Field(default="dummy-key-for-local", alias="LLM_API_KEY")
    llm_prompt_version: str = Field(default="v1", alias="LLM_PROMPT_VERSION")
    llm_max_concurrency: int = Field(default=10, alias="LLM_MAX_CONCURRENCY")
    llm_timeout: float = Field(default=30.0, alias="LLM_TIMEOUT")

    # 上游 ASR 配置
    local_asr_base_url: str = Field(default="http://192.168.1.119:8015", alias="LOCAL_ASR_BASE_URL")
    local_asr_model: str = Field(default="vibevoice", alias="LOCAL_ASR_MODEL")
    local_asr_api_key: str = Field(default="dummy-key-for-local", alias="LOCAL_ASR_API_KEY")
    qwen3_asr_base_url: str = Field(default="http://192.168.1.119:8014", alias="QWEN3_ASR_BASE_URL")
    qwen3_asr_model: str = Field(default="/models/Qwen3-ASR-1.7B", alias="QWEN3_ASR_MODEL")
    qwen3_asr_api_key: str = Field(default="dummy-key-for-local", alias="QWEN3_ASR_API_KEY")

    # 路径配置
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    prompts_dir: Path = Field(default=PROJECT_ROOT / "src" / "prompts", alias="PROMPTS_DIR")
    hotwords_path: Path = Field(default=PROJECT_ROOT / "data" / "lexicon" / "hotwords.json", alias="HOTWORDS_PATH")
    lexicon_dir: Path = Field(default=PROJECT_ROOT / "data" / "lexicon", alias="LEXICON_DIR")

    # 版本配置（用于热词/别名多版本管理，可选）
    hotwords_version: Optional[str] = Field(default=None, alias="HOTWORDS_VERSION")
    aliases_version: Optional[str] = Field(default=None, alias="ALIASES_VERSION")

    # Dify 桥接配置
    dify_enabled: bool = Field(default=False, alias="DIFY_ENABLED")
    dify_base_url: str = Field(default="http://localhost:5001", alias="DIFY_BASE_URL")
    dify_api_key: Optional[str] = Field(default=None, alias="DIFY_API_KEY")
    dify_hotwords_dataset_id: Optional[str] = Field(default=None, alias="DIFY_HOTWORDS_DATASET_ID")
    dify_prompts_dataset_id: Optional[str] = Field(default=None, alias="DIFY_PROMPTS_DATASET_ID")
    dify_aliases_dataset_id: Optional[str] = Field(default=None, alias="DIFY_ALIASES_DATASET_ID")
    dify_knowledge_dataset_id: Optional[str] = Field(default=None, alias="DIFY_KNOWLEDGE_DATASET_ID")
    dify_sync_interval_seconds: int = Field(default=300, alias="DIFY_SYNC_INTERVAL_SECONDS")

    # 功能开关
    enable_entity_guard: bool = Field(default=True, alias="ENABLE_ENTITY_GUARD")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    cache_size: int = Field(default=128, alias="CACHE_SIZE")

    @property
    def prompt_registry_path(self) -> Path:
        return self.prompts_dir / "registry.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
