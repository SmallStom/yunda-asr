"""Layer 2: 词典强制纠错层.

通过领域术语词典进行确定性纠错，适合同音/近音别名替换.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DictionaryCorrectionResult:
    text: str
    original: str
    changes: List[dict] = field(default_factory=list)


class DictionaryCorrector:
    """基于Trie树的词典纠错器."""

    def __init__(self, lexicon_dir: Path | str | None = None):
        if lexicon_dir is None:
            lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
        else:
            lexicon_dir = Path(lexicon_dir)

        self.lexicon_dir = lexicon_dir
        self.alias_map: Dict[str, str] = {}
        self.terms: Dict[str, dict] = {}
        self.patterns: List[tuple[re.Pattern, str]] = []

        self._load_data()

    def _load_data(self) -> None:
        """加载术语库和别名映射."""
        alias_file = self.lexicon_dir / "aliases.json"
        terms_file = self.lexicon_dir / "railway_terms.json"

        if alias_file.exists():
            with open(alias_file, "r", encoding="utf-8") as f:
                self.alias_map = json.load(f)

        if terms_file.exists():
            with open(terms_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for term_info in data.get("terms", []):
                    canonical = term_info["canonical"]
                    self.terms[canonical] = term_info
                    # 编译正则模式
                    for p in term_info.get("patterns", []):
                        if p:
                            try:
                                self.patterns.append((re.compile(p), canonical))
                            except re.error:
                                pass

    def process(self, text: str) -> DictionaryCorrectionResult:
        """执行词典纠错."""
        original = text
        changes = []
        current = text

        # 步骤1：别名精确替换（最长匹配优先）
        current, alias_changes = self._replace_aliases(current)
        changes.extend(alias_changes)

        # 步骤2：正则模式匹配补充
        current, pattern_changes = self._apply_patterns(current)
        changes.extend(pattern_changes)

        return DictionaryCorrectionResult(
            text=current,
            original=original,
            changes=changes,
        )

    def _replace_aliases(self, text: str, whole_word: bool = False) -> tuple[str, List[dict]]:
        """基于别名映射做替换，优先最长匹配，非重叠替换.

        Args:
            text: 输入文本
            whole_word: 是否仅全词匹配（前后非汉字/数字/字母时才算独立词）
        """
        changes = []
        if not self.alias_map:
            return text, changes

        # 按别名长度降序，确保最长匹配优先
        sorted_items = sorted(
            self.alias_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        covered = bytearray(len(text))
        replacements = []  # (pos, alias, canonical)

        for alias, canonical in sorted_items:
            if alias == canonical or not alias:
                continue
            # 仅单字别名（<=1字）自动启用全词匹配保护，避免在长词内部误触
            # 铁路术语中2字及以上别名通常是完整词，无需强制全词匹配
            auto_whole_word = whole_word or len(alias) <= 1
            start = 0
            while True:
                idx = text.find(alias, start)
                if idx == -1:
                    break
                end = idx + len(alias)
                # 检查该区间是否已被更长别名覆盖（bytearray切片比list更快）
                if not any(covered[idx:end]):
                    if auto_whole_word:
                        before = text[idx - 1] if idx > 0 else ''
                        after = text[end] if end < len(text) else ''
                        # 全词边界：前后不能是汉字、数字或字母
                        if (before and re.match(r'[\u4e00-\u9fa5\w]', before)) or \
                           (after and re.match(r'[\u4e00-\u9fa5\w]', after)):
                            start = idx + 1
                            continue
                    replacements.append((idx, alias, canonical))
                    for i in range(idx, end):
                        covered[i] = True
                    start = end
                else:
                    start = idx + 1

        if not replacements:
            return text, changes

        # 按位置降序排序，从后往前替换以避免位置偏移
        replacements.sort(key=lambda x: x[0], reverse=True)

        current = text
        for idx, alias, canonical in replacements:
            current = current[:idx] + canonical + current[idx + len(alias):]
            changes.append({
                "layer": "dictionary",
                "type": "alias_replace",
                "before": alias,
                "after": canonical,
            })

        return current, changes

    def _apply_patterns(self, text: str) -> tuple[str, List[dict]]:
        """应用正则模式做补充纠正（主动替换版）.

        策略：基于标准术语的 pattern 生成包含别名的容错正则，
        对别名变体进行主动替换。仅替换 pattern 中匹配到的别名部分，
        不动数字和量词，确保安全性。
        """
        changes = []
        current = text

        for canonical, term_info in self.terms.items():
            aliases = [a for a in term_info.get("aliases", []) if a != canonical]
            if not aliases:
                continue

            # 构建别名集合，用于快速判断
            alias_set = set(aliases)

            for pattern_str in term_info.get("patterns", []):
                if not pattern_str:
                    continue

                # 生成容错 pattern：将 canonical 替换为 (?:canonical|alias1|alias2|...)
                escaped_canonical = re.escape(canonical)
                if escaped_canonical not in pattern_str:
                    continue

                tolerance_group = "(?:" + "|".join(
                    re.escape(a) for a in ([canonical] + aliases)
                ) + ")"
                tolerance_pattern_str = pattern_str.replace(
                    escaped_canonical, tolerance_group, 1
                )

                try:
                    compiled = re.compile(tolerance_pattern_str)
                except re.error:
                    continue

                # 查找匹配并进行替换
                offset = 0
                for m in compiled.finditer(current):
                    matched = m.group(0)
                    # 如果匹配文本已包含标准词，无需替换
                    # （避免别名是标准词子串时误替换，如 "区间逻辑" 是 "区间逻辑检查" 的子串）
                    if canonical in matched:
                        continue
                    # 检查匹配文本中是否包含别名（而非标准词）
                    # 按长度降序检查，优先匹配更长的别名
                    found_alias = None
                    for alias in sorted(alias_set, key=len, reverse=True):
                        if alias in matched:
                            found_alias = alias
                            break

                    if found_alias is None:
                        continue  # 已经是标准词，无需替换

                    # 仅将别名部分替换为标准词
                    new_matched = matched.replace(found_alias, canonical, 1)
                    if new_matched == matched:
                        continue

                    # 执行替换（考虑前面替换导致的偏移）
                    start = m.start() + offset
                    end = m.end() + offset
                    current = current[:start] + new_matched + current[end:]

                    changes.append({
                        "layer": "dictionary",
                        "type": "pattern_replace",
                        "pattern": tolerance_pattern_str,
                        "before": matched,
                        "after": new_matched,
                    })

                    # 更新偏移量
                    offset += len(new_matched) - len(matched)

        return current, changes

    def reload(self) -> None:
        """热重载词典数据."""
        self.alias_map.clear()
        self.terms.clear()
        self.patterns.clear()
        self._load_data()


# 全局纠错器实例
_corrector = None


def get_dictionary_corrector(lexicon_dir: Path | str | None = None) -> DictionaryCorrector:
    global _corrector
    if _corrector is None:
        _corrector = DictionaryCorrector(lexicon_dir)
    return _corrector


def reload_aliases() -> None:
    """重新加载别名映射（更新全局单例）."""
    global _corrector
    if _corrector is not None:
        _corrector.reload()
