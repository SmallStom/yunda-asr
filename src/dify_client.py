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

    @staticmethod
    def _matches_version(doc_name: str, version: Optional[str]) -> bool:
        """判断文档是否属于指定版本.

        约定：文档名（去扩展名）等于版本名，或以 `{版本}_` 开头。
        例如 version="调度" 匹配 "调度.txt"、"调度_机车.txt"、"调度.json"。
        version 为 None 时匹配所有文档。
        """
        if version is None:
            return True
        stem = Path(doc_name).stem if doc_name else ""
        return stem == version or stem.startswith(f"{version}_")

    @staticmethod
    def _extract_category(doc_name: str, version: Optional[str]) -> str:
        """从文档名提取分类.

        无版本时：用完整 stem 作为分类（向后兼容）。
        有版本时：若 stem == 版本名，分类为 "default"；否则去掉 `{版本}_` 前缀。
        """
        stem = Path(doc_name).stem if doc_name else "default"
        if version is None:
            return stem
        if stem == version:
            return "default"
        if stem.startswith(f"{version}_"):
            return stem[len(version) + 1:]
        return stem

    def fetch_hotwords(self, dataset_id: str, version: Optional[str] = None) -> List[Dict]:
        """从 Dify 知识库拉取热词.

        约定：每个文档为一个分类，文档中每个分片为一行热词。
        文档名作为 category（去掉 .txt 后缀），metadata 中 enabled 可覆盖。

        Args:
            dataset_id: Dify 知识库 ID。
            version: 可选版本名。若指定，仅拉取文档名匹配该版本的文档，
                     例如 version="调度" 会匹配 "调度.txt"、"调度_机车.txt"。
        """
        documents = self.list_documents(dataset_id)
        words = []
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")

            if not self._matches_version(doc_name, version):
                continue

            category = self._extract_category(doc_name, version)

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

        logger.info(f"fetched {len(words)} hotwords from Dify dataset {dataset_id} (version={version})")
        return words

    def fetch_prompts(self, dataset_id: str, version: Optional[str] = None) -> List[Dict]:
        """从 Dify 知识库拉取 Prompt.

        约定：文档名格式为 `{version}_system.txt` 或 `{version}_user_template.txt`。
        version 参数可指定只拉取某个版本，例如 version="调度" 只拉取
        "调度_system.txt" 和 "调度_user_template.txt"。
        """
        documents = self.list_documents(dataset_id)
        prompts = []
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")

            stem = Path(doc_name).stem
            if "_system" in stem:
                prompt_version = stem.replace("_system", "")
                role = "system"
            elif "_user_template" in stem:
                prompt_version = stem.replace("_user_template", "")
                role = "user"
            else:
                continue

            # 版本过滤
            if version is not None and prompt_version != version:
                continue

            content = self.get_document_content(dataset_id, doc_id)
            prompts.append({"version": prompt_version, "role": role, "content": content})

        logger.info(f"fetched {len(prompts)} prompt files from Dify dataset {dataset_id} (version={version})")
        return prompts

    def fetch_aliases(self, dataset_id: str, version: Optional[str] = None) -> Dict[str, str]:
        """从 Dify 知识库拉取正别名映射.

        约定：
        - 文档名为 `aliases.json` 时，内容解析为完整的 JSON 字典。
        - 其他文档每行一个映射，支持 `alias -> canonical` 或 `alias|canonical`。
        - 以 `#` 开头的行为注释。
        - version 参数可指定只拉取某个版本的文档，例如 version="调度"
          会匹配 "调度.json"、"调度.txt"、"调度_*.txt"。
        """
        import json

        documents = self.list_documents(dataset_id)
        aliases: Dict[str, str] = {}
        for doc in documents:
            doc_data = doc.dict() if hasattr(doc, "dict") else doc
            doc_id = doc_data.get("id") or doc_data.get("document", {}).get("id")
            doc_name = doc_data.get("name") or doc_data.get("document", {}).get("name", "")

            # 版本过滤：无版本时匹配所有；有版本时匹配 "调度.json"、"调度.txt"、"调度_*.txt"
            if version is not None:
                stem = Path(doc_name).stem if doc_name else ""
                # 兼容旧格式 aliases.json（无版本前缀）只在无版本过滤时加载
                if stem == "aliases":
                    continue
                if stem != version and not stem.startswith(f"{version}_"):
                    continue

            content = self.get_document_content(dataset_id, doc_id)

            if doc_name.lower().endswith(".json"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        aliases.update(data)
                except Exception:
                    logger.warning(f"failed to parse aliases JSON from {doc_name}")
                continue

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "->" in line:
                    alias, canonical = line.split("->", 1)
                elif "|" in line:
                    alias, canonical = line.split("|", 1)
                else:
                    continue
                alias = alias.strip()
                canonical = canonical.strip()
                if alias and canonical:
                    aliases[alias] = canonical

        logger.info(f"fetched {len(aliases)} aliases from Dify dataset {dataset_id} (version={version})")
        return aliases

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
