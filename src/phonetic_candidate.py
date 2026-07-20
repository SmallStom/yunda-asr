"""拼音混淆候选召回模块.

基于预计算的别名拼音反向索引，快速召回同音/近音候选词。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pypinyin


_singleton_instance: Optional["PhoneticCandidateGenerator"] = None


def get_phonetic_candidate_generator(
    lexicon_dir: Optional[Path] = None,
    confusion_file: Optional[Path] = None,
) -> "PhoneticCandidateGenerator":
    """获取全局单例拼音候选生成器."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = PhoneticCandidateGenerator(
            lexicon_dir=lexicon_dir,
            confusion_file=confusion_file,
        )
    return _singleton_instance


class PhoneticCandidateGenerator:
    """拼音混淆候选生成器."""

    def __init__(
        self,
        lexicon_dir: Optional[Path] = None,
        confusion_file: Optional[Path] = None,
    ):
        if lexicon_dir is None:
            lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
        else:
            lexicon_dir = Path(lexicon_dir)

        self.lexicon_dir = lexicon_dir
        self.confusion_file = confusion_file or (lexicon_dir / "phonetic_confusion.json")

        # 词语混淆映射表（从 data/lexicon/word_confusion.json 加载）
        self.railway_word_confusion: Dict[str, str] = {}
        # 短语级纠错模式（从 data/lexicon/phrase_patterns.json 加载）
        self.phrase_patterns: List[Tuple[re.Pattern, str]] = []

        # 拼音 -> [别名] 的反向索引
        self.pinyin_index: Dict[str, List[str]] = {}
        # 别名 -> 标准词 的映射
        self.alias_to_canonical: Dict[str, str] = {}
        # 混淆规则
        self.confusion: Dict = {}

        self._load_data()
        self._build_index()

    def _load_data(self) -> None:
        """加载词典、混淆规则和纠错配置."""
        # 加载别名映射
        alias_file = self.lexicon_dir / "aliases.json"
        if alias_file.exists():
            with open(alias_file, "r", encoding="utf-8") as f:
                self.alias_to_canonical = json.load(f)

        # 加载拼音混淆规则
        if self.confusion_file.exists():
            with open(self.confusion_file, "r", encoding="utf-8") as f:
                self.confusion = json.load(f)

        # 加载词语混淆映射表
        word_confusion_file = self.lexicon_dir / "word_confusion.json"
        if word_confusion_file.exists():
            with open(word_confusion_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                # 过滤掉以 _ 开头的注释字段
                self.railway_word_confusion = {
                    k: v for k, v in raw.items() if not k.startswith("_")
                }

        # 加载短语级纠错模式
        phrase_patterns_file = self.lexicon_dir / "phrase_patterns.json"
        if phrase_patterns_file.exists():
            with open(phrase_patterns_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                # 过滤掉以 _ 开头的注释字段，编译正则
                self.phrase_patterns = [
                    (re.compile(item["pattern"]), item["replacement"])
                    for item in raw
                    if "pattern" in item and "replacement" in item
                ]

    def _build_index(self) -> None:
        """构建拼音反向索引."""
        self.pinyin_index.clear()
        for alias in self.alias_to_canonical:
            if not alias:
                continue
            py = self._get_pinyin(alias)
            self.pinyin_index.setdefault(py, []).append(alias)

    def reload(self) -> None:
        """重新从磁盘加载别名及混淆规则并重建索引."""
        self._load_data()
        self._build_index()

    def _get_pinyin(self, text: str) -> str:
        """获取文本的拼音序列（无音调，空格分隔）."""
        pys = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
        return " ".join(pys)

    def _apply_confusion(self, py: str) -> List[str]:
        """对拼音应用混淆规则，生成变体拼音列表."""
        variants = [py]
        if not self.confusion:
            return variants

        parts = py.split()
        initial_conf = self.confusion.get("initial", {})
        final_conf = self.confusion.get("final", {})
        whole_conf = self.confusion.get("whole", {})

        # 对每个音节，尝试替换声母/韵母/整音
        for i in range(len(parts)):
            orig = parts[i]
            # 整音混淆（如 si/shi, zi/zhi）
            if orig in whole_conf:
                for variant in whole_conf[orig]:
                    if variant != orig:
                        new_parts = parts.copy()
                        new_parts[i] = variant
                        variants.append(" ".join(new_parts))

            # 声母/韵母混淆需要拆分（简化处理：如果存在以该拼音开头的规则）
            # 由于 pypinyin.Style.NORMAL 输出的是完整拼音，直接做整音混淆更高效
            for key, conf_list in initial_conf.items():
                if orig.startswith(key):
                    for variant in conf_list:
                        if variant != key:
                            new_py = variant + orig[len(key):]
                            if new_py != orig:
                                new_parts = parts.copy()
                                new_parts[i] = new_py
                                variants.append(" ".join(new_parts))

        return list(set(variants))

    def generate_candidates(self, word: str, top_k: int = 5) -> List[dict]:
        """为给定词生成拼音混淆候选.

        返回列表，每项包含：
            - candidate: 候选标准词
            - source: 匹配来源别名
            - pinyin_match: 是否拼音直接匹配
        """
        if not word or len(word) < 2 or len(word) > 6:
            return []

        candidates = []
        seen = set()

        # 铁路场景专用混淆映射优先匹配
        if word in self.railway_word_confusion:
            canonical = self.railway_word_confusion[word]
            seen.add(canonical)
            candidates.append({
                "candidate": canonical,
                "source_alias": word,
                "pinyin_match": False,
            })

        # 如果词本身就在词典中，无需生成拼音候选
        if word in self.alias_to_canonical or word in self.alias_to_canonical.values():
            return candidates[:top_k]

        word_py = self._get_pinyin(word)
        variants = self._apply_confusion(word_py)

        for variant_py in variants:
            for alias in self.pinyin_index.get(variant_py, []):
                canonical = self.alias_to_canonical.get(alias)
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append({
                    "candidate": canonical,
                    "source_alias": alias,
                    "pinyin_match": variant_py == word_py,
                })

        # 按优先级排序：拼音精确匹配优先，其次按别名长度（长别名更具体）
        candidates.sort(key=lambda x: (not x["pinyin_match"], -len(x["source_alias"])))
        return candidates[:top_k]

    def find_candidates_in_text(self, text: str) -> List[dict]:
        """在文本中查找未识别但可能是拼音混淆错误的词，并返回候选.

        返回列表，每项包含：
            - word: 原文中的词
            - position: 位置
            - candidates: 候选列表
        """
        import jieba
        words = list(jieba.cut(text))
        results = []
        offset = 0

        for w in words:
            if len(w) >= 2 and w not in self.alias_to_canonical and w not in self.alias_to_canonical.values():
                cands = self.generate_candidates(w)
                if cands:
                    results.append({
                        "word": w,
                        "position": offset,
                        "candidates": cands,
                    })
            offset += len(w)

        return results


def reload_aliases() -> None:
    """重新加载别名映射（更新全局单例）."""
    global _singleton_instance
    if _singleton_instance is not None:
        _singleton_instance.reload()
