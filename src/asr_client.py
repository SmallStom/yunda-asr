"""ASR 客户端：调用语音转文本服务.

支持 OpenAI 兼容的 /v1/audio/transcriptions 接口，
可对接 Qwen3-ASR (vLLM)、VibeVoice 等本地部署的 ASR 服务。
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx

from src.config import get_settings
from src.logging_config import get_logger


logger = get_logger(__name__)


class ASRClient:
    """ASR 客户端（OpenAI 兼容接口）."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        settings = get_settings()
        self.base_url = (
            base_url
            or os.getenv("ASR_BASE_URL")
            or settings.qwen3_asr_base_url
        ).rstrip("/")
        self.api_key = api_key or os.getenv("ASR_API_KEY") or settings.qwen3_asr_api_key
        self.model = model or os.getenv("ASR_MODEL") or settings.qwen3_asr_model
        self.timeout = timeout

    def transcribe(self, audio_path: str | Path) -> str:
        """将音频文件转为文本.

        Args:
            audio_path: 音频文件路径（wav/mp3/flac 等）。

        Returns:
            识别出的文本。
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        url = f"{self.base_url}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {
                "model": self.model,
                "response_format": "text",
            }

            logger.info(
                "ASR transcribe",
                extra={"url": url, "model": self.model, "file": audio_path.name},
            )

            start = time.perf_counter()
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, files=files, data=data)
                resp.raise_for_status()
                latency = (time.perf_counter() - start) * 1000

            text = resp.text.strip()
            # 某些 ASR 服务忽略 response_format=text，返回 JSON，需要提取 text 字段
            if text.startswith("{"):
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and "text" in data:
                        text = data["text"].strip()
                except json.JSONDecodeError:
                    pass  # 不是合法 JSON，按纯文本处理
            logger.info("ASR done", extra={"latency_ms": latency, "text_len": len(text)})
            return text
