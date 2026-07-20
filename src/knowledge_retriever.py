"""领域知识RAG检索引擎.

三层知识检索：
1. 术语映射检索 - 基于铁路术语库和混淆映射表
2. 错误模式检索 - 基于历史ASR错误对数据
3. 调度场景检索 - 基于预定义场景规则
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba
import pypinyin

from src.phonetic_candidate import PhoneticCandidateGenerator


class KnowledgeRetriever:
    """领域知识检索器，提供三层知识检索能力."""

    # 预定义铁路调度场景规则（场景名 -> 关键术语列表）
    SCENE_RULES: Dict[str, List[str]] = {
        "道岔故障": ["道岔无表示", "现场检查", "抢修", "加锁", "反位"],
        "接发列车": ["接车进路", "发车进路", "引导信号", "开放信号"],
        "设备检查": ["行车设备检查登记簿", "电务销记", "工务销记"],
        "调车作业": ["调车", "进路", "批准"],
        "手摇把作业": ["手摇把", "钩锁器", "转辙机钥匙", "加岗人员", "扳道员"],
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """初始化检索器，加载各类知识数据.

        Args:
            data_dir: 数据目录路径，默认为项目根目录下的 data 目录
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.lexicon_dir = data_dir / "lexicon"
        self.corpus_dir = data_dir / "corpus"

        # 层1数据：铁路术语库
        self.railway_terms: List[dict] = []
        # 从 phonetic_candidate 实例获取铁路混淆映射表
        self._candidate_gen = PhoneticCandidateGenerator()
        self.word_confusion: Dict[str, str] = self._candidate_gen.railway_word_confusion

        # 层2数据：错误模式频率索引
        # 错误词 -> [(正确词, 频率, 上下文)]
        self.error_pattern_index: Dict[str, List[Tuple[str, int, str]]] = {}
        # 所有错误对记录（用于上下文检索）
        self.error_pairs: List[dict] = []

        # 加载数据，失败时降级为空列表
        self._load_railway_terms()
        self._load_error_pairs()

    # ========== 数据加载 ==========

    def _load_railway_terms(self) -> None:
        """加载铁路术语库，失败时降级为空列表."""
        terms_file = self.lexicon_dir / "railway_terms.json"
        if not terms_file.exists():
            return
        try:
            with open(terms_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.railway_terms = data.get("terms", [])
        except Exception:
            self.railway_terms = []

    def _load_error_pairs(self) -> None:
        """加载ASR错误对数据并构建频率索引，失败时降级为空."""
        pairs_file = self.corpus_dir / "asr_error_pairs.jsonl"
        if not pairs_file.exists():
            return
        try:
            pairs: List[dict] = []
            with open(pairs_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self.error_pairs = pairs
            self._build_error_pattern_index()
        except Exception:
            self.error_pairs = []
            self.error_pattern_index = {}

    def _build_error_pattern_index(self) -> None:
        """从错误对数据构建错误模式频率索引.

        对每条记录计算错误模式（asr中有但correct中没有的词）和
        正确模式（correct中有但asr中没有的词），统计高频错误模式。
        """
        # 模式频率统计： (错误词, 正确词) -> 频率
        pattern_freq: Dict[Tuple[str, str], int] = {}
        # 模式上下文： (错误词, 正确词) -> [上下文样本]
        pattern_contexts: Dict[Tuple[str, str], List[str]] = {}

        for pair in self.error_pairs:
            asr_text = pair.get("asr", "")
            correct_text = pair.get("correct", "")
            if not asr_text or not correct_text:
                continue

            # 分词计算差异
            asr_words = set(jieba.lcut(asr_text))
            correct_words = set(jieba.lcut(correct_text))

            # 错误词：asr中有但correct中没有的词
            error_words = asr_words - correct_words
            # 正确词：correct中有但asr中没有的词
            correct_diff_words = correct_words - asr_words

            # 过滤掉标点、单字等无意义词
            error_words = {w for w in error_words if len(w) >= 2 and w.strip()}
            correct_diff_words = {w for w in correct_diff_words if len(w) >= 2 and w.strip()}

            # 构建错误→正确模式对
            for ew in error_words:
                for cw in correct_diff_words:
                    key = (ew, cw)
                    pattern_freq[key] = pattern_freq.get(key, 0) + 1
                    if key not in pattern_contexts:
                        pattern_contexts[key] = []
                    if len(pattern_contexts[key]) < 3:
                        pattern_contexts[key].append(asr_text)

        # 构建错误词 -> [(正确词, 频率, 上下文)] 的索引
        self.error_pattern_index = {}
        for (ew, cw), freq in pattern_freq.items():
            context = pattern_contexts.get((ew, cw), [""])
            # 取第一个上下文样本
            ctx = context[0] if context else ""
            self.error_pattern_index.setdefault(ew, []).append((cw, freq, ctx))

        # 按频率降序排序
        for ew in self.error_pattern_index:
            self.error_pattern_index[ew].sort(key=lambda x: -x[1])

    # ========== 拼音工具 ==========

    def _get_pinyin(self, text: str) -> str:
        """获取文本的拼音序列（无音调，空格分隔）.

        Args:
            text: 输入文本

        Returns:
            无音调拼音字符串，如 "dao cha"
        """
        pys = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
        return " ".join(pys)

    # ========== 层1: 术语映射检索 ==========

    def _match_terms(self, text: str) -> List[dict]:
        """层1：术语映射检索.

        对输入文本分词，检查每个词是否在铁路混淆映射表中，
        返回匹配的术语映射列表。

        Args:
            text: 输入文本

        Returns:
            术语映射列表，每项格式：
            {"error": "到场", "correct": "道岔", "pinyin": "dao cha"}
        """
        if not text:
            return []

        results: List[dict] = []
        seen_errors: Set[str] = set()

        words = list(jieba.cut(text))
        for word in words:
            word = word.strip()
            if not word or word in seen_errors:
                continue
            # 检查词是否在混淆映射中
            if word in self.word_confusion:
                correct = self.word_confusion[word]
                # 跳过错误词和正确词相同的情况
                if word == correct:
                    continue
                seen_errors.add(word)
                pinyin = self._get_pinyin(correct)
                results.append({
                    "error": word,
                    "correct": correct,
                    "pinyin": pinyin,
                })

        return results

    # ========== 层2: 错误模式检索 ==========

    def _match_error_patterns(self, text: str) -> List[dict]:
        """层2：错误模式检索.

        基于历史ASR错误对数据，检索与输入文本相似的历史错误模式。

        Args:
            text: 输入文本

        Returns:
            错误模式列表，每项格式：
            {"pattern": "到场→道岔", "context": "X号到场", "freq": 23}
        """
        if not text or not self.error_pattern_index:
            return []

        results: List[dict] = []
        seen_patterns: Set[str] = set()

        words = list(jieba.cut(text))
        for word in words:
            word = word.strip()
            if not word or word not in self.error_pattern_index:
                continue
            # 检索该错误词对应的所有正确词候选
            for correct_word, freq, context in self.error_pattern_index[word]:
                pattern_key = f"{word}→{correct_word}"
                if pattern_key in seen_patterns:
                    continue
                # 跳过错误词和正确词相同的情况
                if word == correct_word:
                    continue
                seen_patterns.add(pattern_key)
                results.append({
                    "pattern": pattern_key,
                    "context": context,
                    "freq": freq,
                })

        # 按频率降序排序
        results.sort(key=lambda x: -x["freq"])
        return results

    # ========== 层3: 调度场景检索 ==========

    def _match_scenes(self, text: str) -> List[dict]:
        """层3：调度场景检索.

        基于关键词匹配判断当前文本属于哪个调度场景。

        Args:
            text: 输入文本

        Returns:
            场景列表，每项格式：
            {"scene": "道岔故障", "terms": [...], "confidence": 0.8}
        """
        if not text:
            return []

        results: List[dict] = []

        for scene_name, scene_terms in self.SCENE_RULES.items():
            # 统计匹配到的术语数量
            matched_count = 0
            matched_terms: List[str] = []
            for term in scene_terms:
                if term in text:
                    matched_count += 1
                    matched_terms.append(term)

            if matched_count == 0:
                continue

            # 置信度 = 匹配术语数 / 场景总术语数
            confidence = matched_count / len(scene_terms)
            # 至少匹配一个术语才返回，但置信度低于阈值时降低优先级
            results.append({
                "scene": scene_name,
                "terms": scene_terms,
                "matched_terms": matched_terms,
                "confidence": round(confidence, 2),
            })

        # 按置信度降序排序
        results.sort(key=lambda x: -x["confidence"])
        return results

    # ========== 统一检索接口 ==========

    def retrieve(self, text: str, top_k: int = 5) -> List[dict]:
        """检索与输入文本相关的领域知识.

        合并三层检索结果，返回统一格式的知识列表。

        Args:
            text: 输入文本
            top_k: 返回的最大知识条目数

        Returns:
            知识列表，每项格式：
            {"type": "term"|"error_pattern"|"scene", "content": ..., "confidence": ...}
        """
        if not text:
            return []

        # 执行三层检索
        term_results = self._match_terms(text)
        pattern_results = self._match_error_patterns(text)
        scene_results = self._match_scenes(text)

        # 合并为统一格式
        merged: List[dict] = []

        # 术语映射：高置信度
        for item in term_results:
            merged.append({
                "type": "term",
                "content": item,
                "confidence": 0.9,
            })

        # 错误模式：置信度基于频率归一化
        max_freq = max((p["freq"] for p in pattern_results), default=1)
        for item in pattern_results:
            # 频率越高置信度越高，但不超过0.8
            conf = min(0.8, 0.3 + 0.5 * (item["freq"] / max_freq)) if max_freq > 0 else 0.3
            merged.append({
                "type": "error_pattern",
                "content": item,
                "confidence": round(conf, 2),
            })

        # 场景规则：直接使用场景匹配的置信度
        for item in scene_results:
            merged.append({
                "type": "scene",
                "content": item,
                "confidence": item["confidence"],
            })

        # 按置信度降序排序，取top_k
        merged.sort(key=lambda x: -x["confidence"])
        return merged[:top_k]
