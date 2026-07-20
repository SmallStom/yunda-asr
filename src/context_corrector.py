"""Layer 3: 上下文感知纠错层.

整合 N-gram 语言模型、术语共现规则、拼音混淆候选召回，
对 Layer 2 的输出进行上下文消歧和补充纠错。
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import jieba

from src.collocation_rules import get_all_monitored_terms, get_collocation_rule
from src.ngram_model import NgramModel, get_default_ngram_model
from src.phonetic_candidate import PhoneticCandidateGenerator, get_phonetic_candidate_generator


@dataclass
class ContextCorrectionResult:
    text: str
    original: str
    changes: List[dict] = field(default_factory=list)


class ContextCorrector:
    """上下文感知纠错器."""

    # 可疑模式：这些模式提示可能存在需要上下文校验的替换
    SUSPICIOUS_PATTERNS = [
        (r"\d+号(道差|到岔|到差|倒岔)", "道岔"),
        (r"(新号机|信后机)", "信号机"),
        (r"(烟后区|烟喉区|咽喉去)", "咽喉区"),
        (r"(必色分区|闭色分区)", "闭塞分区"),
        (r"(名白|明百)", "明白"),
    ]

    # 分词缓存（避免重复分词）
    _tokenize_cache: Dict[str, List[str]] = {}

    def __init__(
        self,
        ngram_model: Optional[NgramModel] = None,
        candidate_generator: Optional[PhoneticCandidateGenerator] = None,
    ):
        self.ngram = ngram_model or get_default_ngram_model()
        self.candidate_gen = candidate_generator or get_phonetic_candidate_generator()
        self.monitored_terms = get_all_monitored_terms()

        # 为 jieba 加载监控术语，确保不被切散
        for term in self.monitored_terms:
            jieba.add_word(term, freq=10000)

    def _tokenize(self, text: str) -> List[str]:
        """分词（带缓存）."""
        if text in self._tokenize_cache:
            return self._tokenize_cache[text]
        tokens = list(jieba.cut(text))
        tokens = [t for t in tokens if t.strip()]
        self._tokenize_cache[text] = tokens
        return tokens

    def process(self, text: str) -> ContextCorrectionResult:
        """执行上下文感知纠错."""
        original = text
        changes = []
        current = text

        # 步骤0a：短语级纠错（最高优先级，整短语替换）
        current, phrase_changes = self._apply_phrase_patterns(current)
        changes.extend(phrase_changes)

        # 步骤0b：铁路场景高置信度直接纠错（基于混淆表，不依赖N-gram）
        current, railway_changes = self._apply_railway_confusion(current)
        changes.extend(railway_changes)

        # 步骤1：共现规则校验 + 可疑替换检测
        current, collocation_changes = self._apply_collocation_rules(current)
        changes.extend(collocation_changes)

        # 步骤2：拼音混淆候选召回（对未识别词）
        current, phonetic_changes = self._apply_phonetic_candidates(current)
        changes.extend(phonetic_changes)

        return ContextCorrectionResult(
            text=current,
            original=original,
            changes=changes,
        )

    def _apply_phrase_patterns(self, text: str) -> tuple[str, List[dict]]:
        """应用短语级纠错模式.
        
        优先处理整个短语的替换，避免逐词替换导致语义错误。
        """
        changes = []
        current = text

        for pattern, replacement in self.candidate_gen.phrase_patterns:
            if pattern.search(current):
                old_text = current
                current = pattern.sub(replacement, current)
                if current != old_text:
                    changes.append({
                        "layer": "context",
                        "type": "phrase_pattern",
                        "before": old_text,
                        "after": current,
                        "reason": f"短语级纠错: {pattern.pattern} -> {replacement}",
                    })

        return current, changes

    def _apply_railway_confusion(self, text: str) -> tuple[str, List[dict]]:
        """铁路场景高置信度直接纠错.

        基于 PhoneticCandidateGenerator.railway_word_confusion 表，
        对ASR常见误识别词进行直接替换，不依赖N-gram评分。
        适用于短文本或N-gram模型无法有效评分的场景。
        """
        changes = []
        current = text

        confusion_table = self.candidate_gen.railway_word_confusion
        # 按键长度降序匹配，确保最长匹配优先
        sorted_keys = sorted(confusion_table.keys(), key=len, reverse=True)

        for word in sorted_keys:
            if word in current:
                canonical = confusion_table[word]
                current = current.replace(word, canonical, 1)
                changes.append({
                    "layer": "context",
                    "type": "railway_confusion",
                    "before": word,
                    "after": canonical,
                    "reason": "铁路场景高置信度映射",
                })

        return current, changes

    def _apply_collocation_rules(self, text: str) -> tuple[str, List[dict]]:
        """应用共现规则校验 Layer 2 的纠正结果，修正可疑替换."""
        changes = []
        current = text

        tokens = self._tokenize(text)

        for i, token in enumerate(tokens):
            rule = get_collocation_rule(token)
            if rule is None:
                continue

            # 提取前后各2个词作为上下文
            prev_words = tokens[max(0, i - 2):i]
            next_words = tokens[i + 1:min(len(tokens), i + 3)]

            score = rule.check(prev_words, next_words)

            # 如果共现得分低于阈值，标记为可疑
            if score < rule.score_threshold:
                # 尝试查找其他可能的候选（通过拼音召回）
                candidates = self.candidate_gen.generate_candidates(token, top_k=3)
                best_candidate = None
                best_score = score

                for cand_info in candidates:
                    cand = cand_info["candidate"]
                    cand_rule = get_collocation_rule(cand)
                    if cand_rule is None:
                        continue

                    # 替换后重新计算得分
                    test_tokens = tokens.copy()
                    test_tokens[i] = cand
                    test_text = "".join(test_tokens)
                    cand_score = cand_rule.check(
                        test_tokens[max(0, i - 2):i],
                        test_tokens[i + 1:min(len(test_tokens), i + 3)],
                    )

                    # 同时用 N-gram 打分
                    ngram_score_orig = self.ngram.score_term_in_context(text, token)
                    ngram_score_cand = self.ngram.score_term_in_context(test_text, cand)

                    # 综合判断：共现得分提升 或 N-gram 得分显著提升
                    if cand_score > best_score or (
                        cand_score >= best_score and ngram_score_cand > ngram_score_orig * 1.1
                    ):
                        best_candidate = cand
                        best_score = cand_score

                if best_candidate and best_score > score:
                    # 执行替换：定位到第 i 个 token 的精确位置再替换
                    pos = self._find_token_position(current, tokens, i)
                    if pos >= 0:
                        current = current[:pos] + best_candidate + current[pos + len(token):]
                    else:
                        current = current.replace(token, best_candidate, 1)
                    changes.append({
                        "layer": "context",
                        "type": "collocation_fix",
                        "before": token,
                        "after": best_candidate,
                        "reason": f"共现得分 {score:.2f} -> {best_score:.2f}",
                    })
                    # 增量更新 tokens，避免重新对整句分词
                    tokens[i] = best_candidate

        return current, changes

    @staticmethod
    def _find_token_position(text: str, tokens: List[str], target_index: int) -> int:
        """根据分词结果，定位第 target_index 个 token 在原文中的起始位置."""
        pos = 0
        for i, t in enumerate(tokens):
            # 跳过 text 中 pos 位置的空白字符
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if i == target_index:
                # 确认 text[pos:] 以 t 开头
                if text[pos:pos + len(t)] == t:
                    return pos
                # 回退：在附近搜索
                search_start = max(0, pos - len(t))
                idx = text.find(t, search_start)
                return idx
            pos += len(t)
        return -1

    def _apply_phonetic_candidates(self, text: str) -> tuple[str, List[dict]]:
        """对未识别词应用拼音混淆候选召回."""
        changes = []
        current = text

        # 查找未识别但可能是拼音混淆的词
        findings = self.candidate_gen.find_candidates_in_text(text)

        # 性能优化：限制N-gram评分的候选数量
        # 只处理前5个发现，每个只评分top-1候选
        max_findings = 5
        for finding in findings[:max_findings]:
            word = finding["word"]
            candidates = finding["candidates"]
            if not candidates:
                continue

            # 只取第一个候选（已按相似度排序）
            best_candidate = candidates[0]["candidate"]
            if best_candidate == word:
                continue

            # 快速评分：只比较替换前后的N-gram得分
            orig_score = self.ngram.score_text(text)
            test_text = current.replace(word, best_candidate, 1)
            new_score = self.ngram.score_text(test_text)

            # 只有当候选得分显著高于原文时才替换（相对提升 5%）
            if new_score > orig_score * 1.05:
                current = current.replace(word, best_candidate, 1)
                changes.append({
                    "layer": "context",
                    "type": "phonetic_recall",
                    "before": word,
                    "after": best_candidate,
                    "reason": f"N-gram 得分提升: {orig_score:.4f} -> {new_score:.4f}",
                })

        return current, changes
