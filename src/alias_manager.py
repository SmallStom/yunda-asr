"""正别名映射版本管理.

提供别名的多版本保存、列出、切换能力。
版本文件命名：aliases_{version}.json
活跃文件始终为 aliases.json，切换版本即把版本文件复制为活跃文件并热重载。
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.config import get_settings
from src.logging_config import get_logger


logger = get_logger(__name__)


class AliasManager:
    """正别名映射版本管理器."""

    def __init__(self, lexicon_dir: Optional[Path] = None):
        settings = get_settings()
        self.lexicon_dir = lexicon_dir or settings.lexicon_dir
        self.alias_path = self.lexicon_dir / "aliases.json"
        # 启动时若指定了 ALIASES_VERSION，自动激活该版本
        if settings.aliases_version:
            self._activate_version(settings.aliases_version)

    def _versioned_path(self, version: str) -> Path:
        return self.lexicon_dir / f"aliases_{version}.json"

    def _activate_version(self, version: str) -> bool:
        src = self._versioned_path(version)
        if not src.exists():
            logger.warning(f"aliases version file not found: {src}")
            return False
        shutil.copy2(src, self.alias_path)
        logger.info(f"activated aliases version '{version}' from {src}")
        return True

    def list_versions(self) -> List[Dict]:
        """列出所有可用版本."""
        versions = []
        for p in self.lexicon_dir.glob("aliases_*.json"):
            suffix = p.stem[len("aliases_"):]
            versions.append({
                "version": suffix,
                "path": str(p),
                "size": p.stat().st_size,
            })
        return sorted(versions, key=lambda x: x["version"])

    def switch_version(self, version: str) -> bool:
        """切换当前活跃版本并热重载."""
        ok = self._activate_version(version)
        if not ok:
            return False
        # 触发全局热重载
        from src import dictionary_corrector, phonetic_candidate

        phonetic_candidate.reload_aliases()
        dictionary_corrector.reload_aliases()
        # 刷新 API 流水线中的 RAG/Harness TermTool
        try:
            from src.api.dependencies import get_pipeline

            get_pipeline().reload_aliases()
        except Exception:
            pass
        logger.info(f"switched to aliases version '{version}'")
        return True

    def save_as_version(self, version: str, aliases: Dict[str, str]) -> tuple:
        """将别名数据保存为版本文件（不覆盖活跃文件）.

        Returns:
            (path, deleted) - 保存路径和上一次同步操作的数量
        """
        target = self._versioned_path(version)
        # deleted = 上一次同步操作写入的数量
        deleted = getattr(self, "_last_sync_count", 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(aliases, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._last_sync_count = len(aliases)
        logger.info(f"saved aliases as version '{version}' -> {target} (deleted {deleted})")
        return target, deleted

    def save_as_active(self, aliases: Dict[str, str]) -> tuple:
        """覆盖活跃文件并热重载.

        Returns:
            (path, deleted) - 保存路径和上一次同步操作的数量
        """
        # deleted = 上一次同步操作写入的数量
        deleted = getattr(self, "_last_sync_count", 0)
        # 备份
        if self.alias_path.exists():
            backup_dir = self.lexicon_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.alias_path, backup_dir / "aliases.json.bak")
        self.alias_path.write_text(
            json.dumps(aliases, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._last_sync_count = len(aliases)
        # 热重载
        from src import dictionary_corrector, phonetic_candidate

        phonetic_candidate.reload_aliases()
        dictionary_corrector.reload_aliases()
        try:
            from src.api.dependencies import get_pipeline

            get_pipeline().reload_aliases()
        except Exception:
            pass
        return self.alias_path, deleted


_singleton: Optional[AliasManager] = None


def get_alias_manager() -> AliasManager:
    global _singleton
    if _singleton is None:
        _singleton = AliasManager()
    return _singleton
