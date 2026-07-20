"""术语查询工具：根据查询词返回拼音相似的铁路标准术语.

从railway_terms、aliases、word_confusion、hotwords构建拼音索引，
支持精确拼音匹配和模糊拼音相似度匹配。
"""

import json as _json
from pathlib import Path
from typing import Dict, List, Optional

import pypinyin

from src.knowledge_retriever import KnowledgeRetriever


class TermTool:
    """术语查询工具：根据查询词返回拼音相似的铁路标准术语."""

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None):
        self.retriever = retriever or KnowledgeRetriever()
        self._build_pinyin_index()

    def _build_pinyin_index(self):
        """构建拼音→术语的索引."""
        self.pinyin_to_terms: Dict[str, List[str]] = {}

        # 从railway_terms.json加载标准术语
        for term_entry in self.retriever.railway_terms:
            canonical = term_entry.get("canonical", "")
            if not canonical:
                continue
            py = self._get_pinyin(canonical)
            self.pinyin_to_terms.setdefault(py, []).append(canonical)

        # 从aliases加载
        for alias, canonical in self.retriever._candidate_gen.alias_to_canonical.items():
            py = self._get_pinyin(alias)
            self.pinyin_to_terms.setdefault(py, []).append(canonical)
            # 也用canonical的拼音索引
            py_c = self._get_pinyin(canonical)
            self.pinyin_to_terms.setdefault(py_c, []).append(canonical)

        # 从word_confusion加载
        for wrong, correct in self.retriever.word_confusion.items():
            py = self._get_pinyin(correct)
            self.pinyin_to_terms.setdefault(py, []).append(correct)

        # 从hotwords加载（优先使用 HotwordManager，兼容旧格式）
        try:
            from src.hotword_manager import get_hotword_manager

            hotword_manager = get_hotword_manager()
            hotwords = hotword_manager.get_words(enabled_only=True)
        except Exception:
            hotwords_file = Path(__file__).parent.parent / "data" / "lexicon" / "hotwords.json"
            hotwords = []
            if hotwords_file.exists():
                with open(hotwords_file, "r", encoding="utf-8") as f:
                    raw = _json.load(f)
                for hw in raw:
                    if isinstance(hw, str):
                        hotwords.append(hw)
                    elif isinstance(hw, dict):
                        if hw.get("enabled", True):
                            hotwords.append(hw.get("word", ""))

        for hw in hotwords:
            if hw:
                py = self._get_pinyin(hw)
                self.pinyin_to_terms.setdefault(py, []).append(hw)

    def _get_pinyin(self, text: str) -> str:
        """获取文本的拼音序列（无音调，空格分隔）."""
        pys = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
        return " ".join(pys)

    def _pinyin_similarity(self, py1: str, py2: str) -> float:
        """计算两个拼音序列的相似度（基于编辑距离）."""
        if py1 == py2:
            return 1.0
        words1 = py1.split()
        words2 = py2.split()
        if len(words1) != len(words2):
            # 长度不同，用字符级编辑距离
            max_len = max(len(py1), len(py2))
            if max_len == 0:
                return 0.0
            dist = self._edit_distance(py1.replace(" ", ""), py2.replace(" ", ""))
            return 1.0 - dist / max_len
        # 逐音节比较
        match_count = sum(1 for a, b in zip(words1, words2) if a == b)
        return match_count / len(words1)

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
            prev = curr
        return prev[-1]

    def lookup(self, query: str) -> List[dict]:
        """查询与query拼音相似的铁路术语.

        Returns:
            相似术语列表，每项 {"term": "总人解", "pinyin": "zong ren jie", "similarity": 0.8}
        """
        query_py = self._get_pinyin(query)
        results = []

        # 精确拼音匹配
        if query_py in self.pinyin_to_terms:
            for term in self.pinyin_to_terms[query_py]:
                results.append({"term": term, "pinyin": query_py, "similarity": 1.0})

        # 模糊匹配：计算与所有索引中拼音的相似度
        seen = {r["term"] for r in results}
        fuzzy_results = []
        for py, terms in self.pinyin_to_terms.items():
            sim = self._pinyin_similarity(query_py, py)
            # 保留字数相同且相似度>=0.6的结果
            if sim >= 0.6 and len(py.split()) == len(query_py.split()):
                for term in terms:
                    if term not in seen and term != query:
                        fuzzy_results.append({"term": term, "pinyin": py, "similarity": round(sim, 2)})

        # 按相似度排序，取Top3
        fuzzy_results.sort(key=lambda x: x["similarity"], reverse=True)
        results.extend(fuzzy_results[:3])

        return results[:5]  # 最多返回5个
