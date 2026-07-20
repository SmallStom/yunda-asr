"""N-gram模型自动重训练脚本.

合并现有语料与新增语料，重新训练N-gram模型，
通过困惑度对比决定是否替换旧模型.

Usage:
    python scripts/retrain_ngram.py
    python scripts/retrain_ngram.py --corpus data/corpus/railway_corpus.txt --output data/corpus/ngram_model.json
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ngram_model import NgramModel


def parse_args():
    parser = argparse.ArgumentParser(description="N-gram模型重训练")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/railway_corpus.txt"),
        help="语料文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/ngram_model.json"),
        help="模型输出路径",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="N-gram阶数 (默认: 2, bigram)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="保留集比例 (默认: 0.1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制替换，不比较困惑度",
    )
    return parser.parse_args()


def load_corpus(path: Path) -> List[str]:
    """加载语料文件."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_perplexity(model: NgramModel, texts: List[str]) -> float:
    """计算模型在测试集上的困惑度（简化版）.

    使用平均log-probability的指数.
    """
    import jieba

    total_logprob = 0.0
    total_tokens = 0

    for text in texts:
        tokens = list(jieba.cut(text))
        tokens = [t for t in tokens if t.strip()]
        if len(tokens) < model.n:
            continue
        score = model.score_sequence(tokens)
        total_logprob += score
        total_tokens += len(tokens)

    if total_tokens == 0:
        return float("inf")

    avg_logprob = total_logprob / total_tokens
    # 困惑度 = exp(-avg_logprob)
    try:
        ppl = math.exp(-avg_logprob)
    except OverflowError:
        ppl = float("inf")

    return ppl


def main():
    args = parse_args()

    print(f"[INFO] 加载语料: {args.corpus}")
    texts = load_corpus(args.corpus)
    print(f"[INFO] 语料条数: {len(texts)}")

    if len(texts) < 10:
        print("[ERROR] 语料不足，退出")
        sys.exit(1)

    # 随机划分训练集和测试集
    random.seed(42)
    random.shuffle(texts)
    split_idx = int(len(texts) * (1 - args.test_split))
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]

    print(f"[INFO] 训练集: {len(train_texts)} 条, 测试集: {len(test_texts)} 条")

    # 训练新模型
    print(f"[INFO] 训练 {args.n}-gram 模型...")
    new_model = NgramModel(n=args.n)
    new_model.train(train_texts)

    # 计算新模型困惑度
    new_ppl = compute_perplexity(new_model, test_texts)
    print(f"[INFO] 新模型困惑度: {new_ppl:.4f}")

    old_ppl = None
    if args.output.exists() and not args.force:
        try:
            old_model = NgramModel.load(args.output)
            old_ppl = compute_perplexity(old_model, test_texts)
            print(f"[INFO] 旧模型困惑度: {old_ppl:.4f}")
        except Exception as e:
            print(f"[WARN] 无法加载旧模型: {e}")

    # 决定是否替换
    should_replace = True
    if old_ppl is not None and not args.force:
        if new_ppl > old_ppl * 1.1:
            print(f"[WARN] 新模型困惑度劣化超过10% ({new_ppl:.4f} vs {old_ppl:.4f})，保留旧模型")
            should_replace = False
        else:
            print(f"[INFO] 新模型困惑度优于或接近旧模型，执行替换")

    if should_replace:
        # 备份旧模型
        if args.output.exists():
            import time as _time
            backup_path = args.output.with_suffix(f".json.bak.{_time.strftime('%Y%m%d_%H%M%S')}")
            import shutil
            shutil.copy2(args.output, backup_path)
            print(f"[INFO] 旧模型已备份: {backup_path}")

        new_model.save(args.output)
        print(f"[INFO] 新模型已保存: {args.output}")
    else:
        print("[INFO] 未替换模型")


if __name__ == "__main__":
    main()
