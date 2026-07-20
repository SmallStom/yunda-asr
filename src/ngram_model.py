"""领域 N-gram 语言模型.

基于铁路调度语料训练，支持 unigram/bigram/trigram，
采用 Laplace 平滑，纯 Python 实现，无外部依赖。
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba


class NgramModel:
    """轻量 N-gram 语言模型."""

    def __init__(self, n: int = 2):
        if n < 1 or n > 3:
            raise ValueError("n 必须在 1-3 之间")
        self.n = n
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: Set[str] = set()
        self.total_unigrams = 0
        self._loaded = False

    def train(self, texts: List[str], custom_words: Optional[List[str]] = None) -> None:
        """训练模型：对 texts 分词后统计 n-gram 频次.

        Args:
            texts: 训练语料列表（已预处理的文本）
            custom_words: 需要保护不被切散的自定义词汇列表
        """
        if custom_words:
            for w in custom_words:
                jieba.add_word(w, freq=10000)

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()
        self.total_unigrams = 0

        for text in texts:
            tokens = list(jieba.cut(text))
            tokens = [t for t in tokens if t.strip()]
            if not tokens:
                continue

            # unigram
            for t in tokens:
                self.vocab.add(t)
                self.ngram_counts[(t,)] += 1
                self.total_unigrams += 1

            # bigram
            if self.n >= 2:
                for i in range(len(tokens) - 1):
                    bigram = (tokens[i], tokens[i + 1])
                    self.ngram_counts[bigram] += 1
                    self.context_counts[(tokens[i],)] += 1

            # trigram
            if self.n >= 3:
                for i in range(len(tokens) - 2):
                    trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                    self.ngram_counts[trigram] += 1
                    self.context_counts[(tokens[i], tokens[i + 1])] += 1

        self._loaded = True

    def _get_count(self, ngram: Tuple[str, ...]) -> int:
        return self.ngram_counts.get(ngram, 0)

    def _get_context_count(self, context: Tuple[str, ...]) -> int:
        return self.context_counts.get(context, 0)

    def _laplace_prob(self, ngram: Tuple[str, ...]) -> float:
        """计算单个 n-gram 的 Laplace 平滑概率."""
        count = self._get_count(ngram)
        if len(ngram) == 1:
            # unigram
            return (count + 1) / (self.total_unigrams + len(self.vocab))
        else:
            context = ngram[:-1]
            context_count = self._get_context_count(context)
            return (count + 1) / (context_count + len(self.vocab))

    def score_sequence(self, tokens: List[str]) -> float:
        """计算序列的 log-probability（Laplace 平滑）.

        返回值为负数，绝对值越小表示概率越高（越"像"领域语料）。
        """
        if not tokens:
            return 0.0

        log_prob = 0.0
        V = len(self.vocab)

        for i in range(len(tokens)):
            if self.n == 1 or i == 0:
                # unigram
                count = self._get_count((tokens[i],))
                prob = (count + 1) / (self.total_unigrams + V)
            elif self.n >= 2 and i == 1:
                # bigram（回退到 unigram 如果 bigram 不存在）
                bigram = (tokens[i - 1], tokens[i])
                count = self._get_count(bigram)
                context_count = self._get_count((tokens[i - 1],))
                if count > 0:
                    prob = (count + 1) / (context_count + V)
                else:
                    # 回退
                    count = self._get_count((tokens[i],))
                    prob = (count + 1) / (self.total_unigrams + V)
            else:
                # trigram 优先，其次 bigram，最后 unigram
                if self.n >= 3:
                    trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
                    count = self._get_count(trigram)
                    context_count = self._get_count((tokens[i - 2], tokens[i - 1]))
                    if count > 0:
                        prob = (count + 1) / (context_count + V)
                    else:
                        # 回退 bigram
                        bigram = (tokens[i - 1], tokens[i])
                        count = self._get_count(bigram)
                        context_count = self._get_count((tokens[i - 1],))
                        if count > 0:
                            prob = (count + 1) / (context_count + V)
                        else:
                            count = self._get_count((tokens[i],))
                            prob = (count + 1) / (self.total_unigrams + V)
                else:
                    bigram = (tokens[i - 1], tokens[i])
                    count = self._get_count(bigram)
                    context_count = self._get_count((tokens[i - 1],))
                    prob = (count + 1) / (context_count + V)

            log_prob += math.log(prob)

        return log_prob

    def score_text(self, text: str) -> float:
        """对整句文本打分（内部自动分词）."""
        tokens = list(jieba.cut(text))
        tokens = [t for t in tokens if t.strip()]
        return self.score_sequence(tokens)

    def score_term_in_context(self, text: str, term: str, window: int = 3) -> float:
        """计算术语在其上下文窗口内的局部得分.

        用于比较同一位置不同候选术语的合理性。
        """
        tokens = list(jieba.cut(text))
        tokens = [t for t in tokens if t.strip()]

        try:
            idx = tokens.index(term)
        except ValueError:
            # 术语不在分词结果中，尝试模糊匹配
            return self.score_text(text)

        start = max(0, idx - window)
        end = min(len(tokens), idx + window + 1)
        local_tokens = tokens[start:end]
        return self.score_sequence(local_tokens)

    def save(self, path: Path | str) -> None:
        """保存模型到 JSON 文件."""
        data = {
            "n": self.n,
            "vocab": list(self.vocab),
            "total_unigrams": self.total_unigrams,
            "ngram_counts": {json.dumps(k, ensure_ascii=False): v for k, v in self.ngram_counts.items()},
            "context_counts": {json.dumps(k, ensure_ascii=False): v for k, v in self.context_counts.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "NgramModel":
        """从 JSON 文件加载模型."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(n=data["n"])
        model.vocab = set(data["vocab"])
        model.total_unigrams = data["total_unigrams"]
        model.ngram_counts = {
            tuple(json.loads(k)): v for k, v in data["ngram_counts"].items()
        }
        model.context_counts = {
            tuple(json.loads(k)): v for k, v in data["context_counts"].items()
        }
        model._loaded = True
        return model


def get_default_ngram_model(corpus_path: Optional[Path] = None) -> NgramModel:
    """加载或训练默认 N-gram 模型."""
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "railway_corpus.txt"

    model_path = corpus_path.parent / "ngram_model.json"
    if model_path.exists():
        try:
            return NgramModel.load(model_path)
        except Exception:
            pass

    # 训练新模型
    if not corpus_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    model = NgramModel(n=2)
    # 加载术语库中的词汇作为自定义词，防止被切散
    terms_file = corpus_path.parent.parent / "lexicon" / "railway_terms.json"
    custom_words = []
    if terms_file.exists():
        import json
        with open(terms_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for term_info in data.get("terms", []):
            custom_words.append(term_info["canonical"])
            custom_words.extend(term_info.get("aliases", []))

    model.train(texts, custom_words=list(set(custom_words)))
    model.save(model_path)
    return model
