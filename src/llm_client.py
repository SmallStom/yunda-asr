"""LLM 客户端封装.

支持本地私有化部署的 OpenAI 兼容接口，含同步/异步调用与并发控制。
"""

import asyncio
import os
import time
from typing import Optional

from src.config import get_settings


class LLMClient:
    """本地私有化 LLM 客户端（OpenAI 兼容接口）."""

    DEFAULT_BASE_URL = "http://192.168.1.119:8012/v1"
    DEFAULT_MODEL = "Qwen3.6-27B"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3

    # 全局 LLM 并发信号量
    _semaphore: Optional[asyncio.Semaphore] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        settings = get_settings()
        self.base_url = base_url or os.getenv("LLM_BASE_URL", self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("LLM_API_KEY", "dummy-key-for-local")
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout or settings.llm_timeout or self.DEFAULT_TIMEOUT
        self.max_concurrency = settings.llm_max_concurrency
        self._sync_client = None
        self._async_client = None

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        """获取全局并发信号量."""
        if cls._semaphore is None:
            settings = get_settings()
            cls._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        return cls._semaphore

    def _get_sync_client(self):
        """延迟初始化同步 openai 客户端."""
        if self._sync_client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "请先安装 openai 库: pip install openai"
                ) from e
            self._sync_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self):
        """延迟初始化异步 openai 客户端."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "请先安装 openai 库: pip install openai"
                ) from e
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步调用，带重试机制（指数退避）."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.complete_multi_turn(messages, temperature=0.1, enable_thinking=False)

    def complete_multi_turn(
        self,
        messages: list,
        temperature: float = 0.1,
        enable_thinking: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        """多轮对话调用，带重试机制（指数退避）.

        Args:
            messages: 完整的对话消息列表（含system/user/assistant角色）
            temperature: 采样温度
            enable_thinking: 是否开启Qwen3思考模式
            max_tokens: 最大输出token数
        """
        # 思考模式需要更多token（思考过程+最终输出）
        if enable_thinking and max_tokens < 4096:
            max_tokens = 4096

        client = self._get_sync_client()
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                content = response.choices[0].message.content
                if not content:
                    # 检查是否因token长度限制导致空内容
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "length":
                        # 增大max_tokens后重试
                        max_tokens = min(max_tokens * 2, 8192)
                        raise ValueError(f"LLM因token限制返回空内容(finish_reason=length)，重试max_tokens={max_tokens}")
                    raise ValueError("LLM 返回空内容")
                return content.strip()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise last_error or RuntimeError("LLM 调用失败")

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """异步调用，带并发信号量控制."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.acomplete_multi_turn(messages, temperature=0.1, enable_thinking=False)

    async def acomplete_multi_turn(
        self,
        messages: list,
        temperature: float = 0.1,
        enable_thinking: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        """异步多轮对话调用，带并发信号量与重试."""
        if enable_thinking and max_tokens < 4096:
            max_tokens = 4096

        async with self._get_semaphore():
            client = self._get_async_client()
            last_error = None

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    )
                    content = response.choices[0].message.content
                    if not content:
                        finish_reason = response.choices[0].finish_reason
                        if finish_reason == "length":
                            max_tokens = min(max_tokens * 2, 8192)
                            raise ValueError(f"LLM因token限制返回空内容(finish_reason=length)，重试max_tokens={max_tokens}")
                        raise ValueError("LLM 返回空内容")
                    return content.strip()
                except Exception as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)

        raise last_error or RuntimeError("LLM 调用失败")
