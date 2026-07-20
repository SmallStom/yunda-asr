"""热词管理器.

支持热词的增删改查、持久化、备份与热重载。
兼容旧格式（字符串列表）与新格式（对象数组）。
"""

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from src.config import get_settings
from src.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class HotwordItem:
    """热词项."""

    id: str
    word: str
    category: Optional[str] = None
    enabled: bool = True


class HotwordManager:
    """热词管理器."""

    def __init__(self, hotwords_path: Optional[Path] = None):
        self.settings = get_settings()
        self.hotwords_path = hotwords_path or self.settings.hotwords_path
        self._items: Dict[str, HotwordItem] = {}
        self._lock = Lock()
        # 启动时若指定了 HOTWORDS_VERSION，自动激活该版本
        if self.settings.hotwords_version:
            self._activate_version(self.settings.hotwords_version)
        self.reload()

    def _versioned_path(self, version: str) -> Path:
        """获取版本化文件路径，如 hotwords_v2.json."""
        return self.hotwords_path.with_name(
            self.hotwords_path.stem + f"_{version}" + self.hotwords_path.suffix
        )

    def _activate_version(self, version: str) -> bool:
        """将版本文件复制为当前活跃文件."""
        src = self._versioned_path(version)
        if not src.exists():
            logger.warning(f"hotwords version file not found: {src}")
            return False
        self.hotwords_path.write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
        logger.info(f"activated hotwords version '{version}' from {src}")
        return True

    def list_versions(self) -> List[Dict]:
        """列出所有可用的热词版本."""
        versions = []
        for p in self.hotwords_path.parent.glob(
            self.hotwords_path.stem + "_*" + self.hotwords_path.suffix
        ):
            # 提取版本名
            suffix = p.stem[len(self.hotwords_path.stem) + 1:]
            versions.append({
                "version": suffix,
                "path": str(p),
                "size": p.stat().st_size,
            })
        return sorted(versions, key=lambda x: x["version"])

    def switch_version(self, version: str) -> bool:
        """切换当前活跃版本."""
        with self._lock:
            ok = self._activate_version(version)
            if ok:
                # 重新加载
                try:
                    data = json.loads(self.hotwords_path.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        data = []
                    items = self._migrate_legacy_format(data)
                    self._items = {item.id: item for item in items}
                    logger.info(f"switched to hotwords version '{version}', {len(self._items)} words loaded")
                except Exception as e:
                    logger.error(f"failed to reload after version switch: {e}")
            return ok

    def save_as_version(self, version: str, data: list) -> Path:
        """将数据保存为版本文件."""
        target = self._versioned_path(version)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"saved hotwords as version '{version}' -> {target}")
        return target

    def _ensure_file(self) -> None:
        """确保文件存在."""
        if not self.hotwords_path.exists():
            self.hotwords_path.parent.mkdir(parents=True, exist_ok=True)
            self.hotwords_path.write_text("[]", encoding="utf-8")

    def _migrate_legacy_format(self, data: list) -> List[HotwordItem]:
        """迁移旧格式（字符串列表）到新格式."""
        items = []
        for entry in data:
            if isinstance(entry, str):
                items.append(
                    HotwordItem(
                        id=str(uuid.uuid4()),
                        word=entry,
                        category=None,
                        enabled=True,
                    )
                )
            elif isinstance(entry, dict):
                items.append(
                    HotwordItem(
                        id=entry.get("id") or str(uuid.uuid4()),
                        word=entry["word"],
                        category=entry.get("category"),
                        enabled=entry.get("enabled", True),
                    )
                )
        return items

    def reload(self) -> None:
        """从文件重载热词."""
        with self._lock:
            self._ensure_file()
            try:
                data = json.loads(self.hotwords_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"failed to parse hotwords file: {e}")
                data = []

            if not isinstance(data, list):
                data = []

            items = self._migrate_legacy_format(data)
            self._items = {item.id: item for item in items}
            logger.info(f"loaded {len(self._items)} hotwords from {self.hotwords_path}")

    def _backup(self) -> None:
        """同步前备份原文件."""
        if self.hotwords_path.exists():
            backup_path = self.hotwords_path.with_suffix(".json.bak")
            shutil.copy2(self.hotwords_path, backup_path)

    def _save(self) -> None:
        """保存到文件."""
        self._backup()
        items = sorted(
            [asdict(item) for item in self._items.values()],
            key=lambda x: x["word"],
        )
        self.hotwords_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_all(self, category: Optional[str] = None, enabled_only: bool = True) -> List[HotwordItem]:
        """列出热词."""
        with self._lock:
            items = list(self._items.values())
            if category:
                items = [item for item in items if item.category == category]
            if enabled_only:
                items = [item for item in items if item.enabled]
            return sorted(items, key=lambda x: x.word)

    def get_words(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """获取热词字符串列表."""
        return [item.word for item in self.list_all(category=category, enabled_only=enabled_only)]

    def get(self, hotword_id: str) -> Optional[HotwordItem]:
        """根据 ID 获取热词."""
        with self._lock:
            return self._items.get(hotword_id)

    def create(self, word: str, category: Optional[str] = None, enabled: bool = True) -> HotwordItem:
        """创建热词."""
        with self._lock:
            # 去重：相同 word 且同 category 视为重复
            for item in self._items.values():
                if item.word == word and item.category == category:
                    raise ValueError(f"hotword '{word}' already exists")

            item = HotwordItem(
                id=str(uuid.uuid4()),
                word=word,
                category=category,
                enabled=enabled,
            )
            self._items[item.id] = item
            self._save()
            return item

    def update(self, hotword_id: str, **kwargs) -> Optional[HotwordItem]:
        """更新热词."""
        with self._lock:
            item = self._items.get(hotword_id)
            if not item:
                return None

            if "word" in kwargs:
                item.word = kwargs["word"]
            if "category" in kwargs:
                item.category = kwargs["category"]
            if "enabled" in kwargs:
                item.enabled = kwargs["enabled"]

            self._items[item.id] = item
            self._save()
            return item

    def delete(self, hotword_id: str) -> bool:
        """删除热词."""
        with self._lock:
            if hotword_id not in self._items:
                return False
            del self._items[hotword_id]
            self._save()
            return True

    def reload_from_dify(self, words: List[Dict], version: Optional[str] = None) -> Dict[str, int]:
        """从 Dify 同步热词.

        Args:
            words: 热词对象列表，每项至少包含 word，可选 category/enabled。
            version: 若指定，保存为版本文件（hotwords_{version}.json），
                     不覆盖当前活跃文件；需手动 switch_version 激活。

        Returns:
            {"updated": int, "deleted": int, "skipped": int}
        """
        with self._lock:
            # 先在内存中构建新数据
            new_items: Dict[str, HotwordItem] = {}
            updated = 0
            skipped = 0

            for entry in words:
                word = entry.get("word")
                if not word:
                    skipped += 1
                    continue

                existing_id = None
                for item in self._items.values():
                    if item.word == word and item.category == entry.get("category"):
                        existing_id = item.id
                        break

                item_id = existing_id or str(uuid.uuid4())
                new_items[item_id] = HotwordItem(
                    id=item_id,
                    word=word,
                    category=entry.get("category"),
                    enabled=entry.get("enabled", True),
                )
                updated += 1

            deleted = len(self._items) - len(
                {k for k in self._items if k in new_items}
            )

            # 序列化数据
            serialized = sorted(
                [asdict(item) for item in new_items.values()],
                key=lambda x: x["word"],
            )

            if version:
                # 仅保存为版本文件，不动活跃文件
                self.save_as_version(version, serialized)
            else:
                # 覆盖活跃文件
                self._backup()
                self.hotwords_path.write_text(
                    json.dumps(serialized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._items = new_items

            return {"updated": updated, "deleted": deleted, "skipped": skipped}

    def to_asr_format(self) -> Dict:
        """转换为上期 ASR 服务可接收的格式.

        返回 {"hotwords": [...], "categories": {...}}，具体格式可按 ASR 协议调整。
        """
        words = self.get_words(enabled_only=True)
        categories: Dict[str, List[str]] = {}
        for item in self.list_all(enabled_only=True):
            cat = item.category or "default"
            categories.setdefault(cat, []).append(item.word)
        return {"hotwords": words, "categories": categories}


# 全局单例
_hotword_manager: Optional[HotwordManager] = None


def get_hotword_manager() -> HotwordManager:
    """获取热词管理器单例."""
    global _hotword_manager
    if _hotword_manager is None:
        _hotword_manager = HotwordManager()
    return _hotword_manager
