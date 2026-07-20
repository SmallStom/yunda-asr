"""Dify 集成客户端.

封装 Dify Dataset API，用于从 Dify 知识库拉取热词、Prompt、领域知识。
优先使用 dify-dataset-sdk；若不可用则使用 httpx 原生调用。
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.config import get_settings
from src.logging_config import get_logger


logger = get_logger(__name__)


try:
    from dify_dataset_sdk import DifyDatasetClient

    _DIFY_SDK_AVAILABLE = True
except Exception:
    _DIFY_SDK_AVAILABLE = False


class DifyClientError(Exception):
    """Dify 客户端错误."""


class DifyClient:
    """Dify 数据集客户端封装."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.dify_base_url).rstrip("/")
        self.api_key = api_key or settings.dify_api_key
        self.timeout = settings.request_timeout
        self._client = None

        if not self.api_key:
            raise DifyClientError("DIFY_API_KEY is not configured")

    def _get_sdk_client(self):
        if not _DIFY_SDK_AVAILABLE:
            raise DifyClientError("dify-dataset-sdk is not installed, run: pip install -e '.[dify']")
        if self._client is None:
            self._client = DifyDatasetClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def list_documents(self, dataset_id: str, limit: int = 100) -> List[Dict]:
        """列出知识库下所有文档."""
        client = self._get_sdk_client()
        documents = []
        page = 1
        while True:
            resp = client.documents.list(dataset_id=dataset_id, page=page, limit=limit)
            data = getattr(resp, "data", resp)
            if not data:
                break
            documents.extend(data)
            if len(data) < limit:
                break
            page += 1
        return documents

    def get_document_segments(self, dataset_id: str, document_id: str, limit: int = 100) -> List[Dict]:
        """获取文档的所有分片内容."""
        client = self._get_sdk_client()
        segments = []
        page = 1
        while True:
            resp = client.segments.list(
                dataset_id=dataset_id,
                document_id=document_id,
                page=page,
                limit=limit,
            )
            data = getattr(resp, "data", resp)
            if not data:
                break
            segments.extend(data)
            if len(data) < limit:
                break
            page += 1
        return segments

    def get_document_content(self, dataset_id: str, document_id: str) -> str:
        """获取文档完整内容（拼接所有分片）."""
        segments = self.get_document_segments(dataset_id, document_id)
        parts = []
        for seg in segments:
            seg_data = seg
            if hasattr(seg, "dict"):
                seg_data = seg.dict()
            content = seg_data.get("content", "") if isinstance(seg_data, dict) else getattr(seg, "content", "")
            if content:
                parts.append(content)
        return "\n".join(parts)

    def fetch_hotwords(self, dataset_id: str) -> List[Dict]:
        """从 Dify 知识库拉取热词.

        约定：每个文档为一个分类，文档中每个分片为一行热词。
        文档名作为 category（去掉 .txt 后缀），metadata 中 enabled 可覆盖。
        """
        documents = self.list_documents(dataset_id)
        words = []
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")
            category = Path(doc_name).stem if doc_name else "default"

            content = self.get_document_content(dataset_id, doc_id)
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 支持 JSON 数组或单行一词
                if line.startswith("["):
                    try:
                        import json
                        items = json.loads(line)
                        for item in items:
                            words.append({"word": item, "category": category, "enabled": True})
                        continue
                    except Exception:
                        pass
                words.append({"word": line, "category": category, "enabled": True})

        logger.info(f"fetched {len(words)} hotwords from Dify dataset {dataset_id}")
        return words

    def fetch_prompts(self, dataset_id: str) -> List[Dict]:
        """从 Dify 知识库拉取 Prompt.

        约定：文档名格式为 `{version}_system.txt` 或 `{version}_user_template.txt`。
        """
        documents = self.list_documents(dataset_id)
        prompts = []
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")

            stem = Path(doc_name).stem
            if "_system" in stem:
                version = stem.replace("_system", "")
                role = "system"
            elif "_user_template" in stem:
                version = stem.replace("_user_template", "")
                role = "user"
            else:
                continue

            content = self.get_document_content(dataset_id, doc_id)
            prompts.append({"version": version, "role": role, "content": content})

        logger.info(f"fetched {len(prompts)} prompt files from Dify dataset {dataset_id}")
        return prompts

    def fetch_knowledge(self, dataset_id: str) -> List[Dict]:
        """从 Dify 知识库拉取领域知识/错误模式.

        返回文档列表，每项包含 name 和 content。
        """
        documents = self.list_documents(dataset_id)
        results = []
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")
            content = self.get_document_content(dataset_id, doc_id)
            results.append({"name": doc_name, "content": content})

        logger.info(f"fetched {len(results)} knowledge documents from Dify dataset {dataset_id}")
        return results

    def close(self) -> None:
        """关闭客户端连接."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"failed to close Dify client: {e}")
            self._client = None
