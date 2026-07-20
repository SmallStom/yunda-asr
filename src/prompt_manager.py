"""Prompt 版本管理器.

管理 src/prompts/registry.json 与各版本 system/user_template 文件。
支持运行时切换默认版本、创建/更新版本、从 Dify 同步。
"""

import json
import shutil
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from src.config import get_settings
from src.logging_config import get_logger


logger = get_logger(__name__)


class PromptManager:
    """Prompt 版本管理器."""

    def __init__(self):
        self.settings = get_settings()
        self.prompts_dir = self.settings.prompts_dir
        self.registry_path = self.settings.prompt_registry_path
        self._registry: Dict = {}
        self._lock = Lock()
        self.reload()

    def _load_registry(self) -> Dict:
        """加载 registry.json."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            default_registry = {
                "versions": {},
                "default": "v1",
                "evaluations": {},
            }
            self.registry_path.write_text(
                json.dumps(default_registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return default_registry

        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"failed to parse prompt registry: {e}")
            return {"versions": {}, "default": "v1", "evaluations": {}}

    def reload(self) -> None:
        """重载 Prompt 注册表."""
        with self._lock:
            self._registry = self._load_registry()
            logger.info(f"loaded prompt registry with versions: {list(self._registry.get('versions', {}).keys())}")

    def _save_registry(self) -> None:
        """保存 registry.json."""
        self.registry_path.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_versions(self) -> List[Dict]:
        """列出所有版本."""
        with self._lock:
            versions = self._registry.get("versions", {})
            default = self._registry.get("default", "v1")
            return [
                {
                    "version": version,
                    "description": info.get("description", ""),
                    "created_at": info.get("created_at", ""),
                    "is_default": version == default,
                }
                for version, info in versions.items()
            ]

    def get_version_path(self, version: str) -> Tuple[Path, Path]:
        """获取某版本的 system/user_template 路径."""
        version_dir = self.prompts_dir / version
        return version_dir / "system.txt", version_dir / "user_template.txt"

    def get(self, version: str) -> Optional[Dict]:
        """获取某版本内容."""
        with self._lock:
            versions = self._registry.get("versions", {})
            if version not in versions:
                return None

            system_path, template_path = self.get_version_path(version)
            system = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
            user_template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

            return {
                "version": version,
                "system": system,
                "user_template": user_template,
                "description": versions[version].get("description", ""),
                "created_at": versions[version].get("created_at", ""),
                "is_default": version == self._registry.get("default"),
            }

    def set_default(self, version: str) -> bool:
        """设置默认版本."""
        with self._lock:
            versions = self._registry.get("versions", {})
            if version not in versions:
                return False
            self._registry["default"] = version
            self._save_registry()
            logger.info(f"set default prompt version to {version}")
            return True

    def create_version(
        self,
        version: str,
        system: str,
        user_template: str,
        description: str = "",
        created_at: str = "",
    ) -> bool:
        """创建新版本."""
        with self._lock:
            versions = self._registry.setdefault("versions", {})
            if version in versions:
                return False

            version_dir = self.prompts_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)

            (version_dir / "system.txt").write_text(system, encoding="utf-8")
            (version_dir / "user_template.txt").write_text(user_template, encoding="utf-8")

            versions[version] = {
                "system": f"{version}/system.txt",
                "user_template": f"{version}/user_template.txt",
                "description": description,
                "created_at": created_at,
            }
            self._save_registry()
            logger.info(f"created prompt version {version}")
            return True

    def update_version(self, version: str, system: Optional[str] = None, user_template: Optional[str] = None, description: Optional[str] = None) -> bool:
        """更新版本内容."""
        with self._lock:
            versions = self._registry.get("versions", {})
            if version not in versions:
                return False

            system_path, template_path = self.get_version_path(version)
            if system is not None:
                system_path.write_text(system, encoding="utf-8")
            if user_template is not None:
                template_path.write_text(user_template, encoding="utf-8")
            if description is not None:
                versions[version]["description"] = description

            self._save_registry()
            logger.info(f"updated prompt version {version}")
            return True

    def reload_from_dify(self, prompts: List[Dict]) -> List[str]:
        """从 Dify 同步 Prompt.

        Args:
            prompts: Prompt 对象列表，每项包含 version、role、content。

        Returns:
            已更新的文件路径列表。
        """
        with self._lock:
            updated = []
            for entry in prompts:
                version = entry.get("version")
                role = entry.get("role")
                content = entry.get("content")
                if not version or role not in ("system", "user"):
                    continue

                version_dir = self.prompts_dir / version
                version_dir.mkdir(parents=True, exist_ok=True)

                if role == "system":
                    file_path = version_dir / "system.txt"
                    self._registry.setdefault("versions", {}).setdefault(version, {})["system"] = f"{version}/system.txt"
                else:
                    file_path = version_dir / "user_template.txt"
                    self._registry.setdefault("versions", {}).setdefault(version, {})["user_template"] = f"{version}/user_template.txt"

                file_path.write_text(content, encoding="utf-8")
                updated.append(str(file_path))

            self._save_registry()
            logger.info(f"reloaded {len(updated)} prompt files from Dify")
            return updated


# 全局单例
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """获取 Prompt 管理器单例."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
