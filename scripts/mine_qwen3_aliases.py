"""挖掘 qwen3-asr 原始输出中的高频近音误识别候选，用于补充 word_confusion.json.

筛选原则：
- 长度 2-8 字，出现 >=2 次
- 拼音（无声调）相同或编辑距离 <=1
- 非子串归一化（避免语义归一化）
- 不与现有 aliases/word_confusion 重复
- 排除危险映射（具体术语→大类）
"""

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from pypinyin import lazy_pinyin

LEXICON = Path("data/lexicon")
DANGEROUS = {"进站信号机", "出站信号机", "通过信号机", "色灯信号机", "信号灯", "进站", "出站"}


def load_existing():
    aliases = json.loads((LEXICON / "aliases.json").read_text(encoding="utf-8"))
    word_conf = json.loads((LEXICON / "word_confusion.json").read_text(encoding="utf-8"))
    existing_keys = set(aliases.keys()) | set(word_conf.keys())
    # 也排除已存在的标准词，避免反向
    existing_vals = set(aliases.values()) | set(word_conf.values())
    return existing_keys, existing_vals


def pinyin_str(s: str) -> str:
    return "".join(lazy_pinyin(s))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        ndp = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[lb]


def is_phonetic_similar(alias: str, canonical: str) -> bool:
    if alias == canonical:
        return False
    # 子串归一化
    if (alias in canonical or canonical in alias) and abs(len(alias) - len(canonical)) > 1:
        return False
    pa = pinyin_str(alias)
    pc = pinyin_str(canonical)
    if pa == pc:
        return True
    # 长度差过大不认为是近音
    if abs(len(pa) - len(pc)) > 2:
        return False
    return edit_distance(pa, pc) <= 1


def extract_pairs(asr_text: str, correct_text: str):
    pairs = []
    matcher = SequenceMatcher(None, asr_text, correct_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            a = asr_text[i1:i2]
            c = correct_text[j1:j2]
            if len(a) >= 2 and len(c) >= 2:
                pairs.append((a, c))
    return pairs


def main():
    existing_keys, existing_vals = load_existing()
    testset = Path("data/asr_testset/asr_test_pairs_qwen3.jsonl")
    records = [json.loads(l) for l in testset.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[INFO] 加载 {len(records)} 条 qwen3 平行语对")

    counter = Counter()
    for r in records:
        for a, c in extract_pairs(r["asr"], r["correct"]):
            counter[(a, c)] += 1

    candidates = []
    for (a, c), n in counter.most_common():
        if n < 2:
            continue
        if not (2 <= len(a) <= 8 and 2 <= len(c) <= 8):
            continue
        if a in existing_keys or c in DANGEROUS:
            continue
        if not is_phonetic_similar(a, c):
            continue
        candidates.append((a, c, n, pinyin_str(a), pinyin_str(c)))

    print(f"\n高频近音候选（出现>=2次，拼音相同或编辑距离<=1）: {len(candidates)} 条")
    print(f"{'ASR片段':<12s} {'正确片段':<12s} {'次数':>4s} {'ASR拼音':<12s} {'正确拼音':<12s}")
    print("-" * 60)
    for a, c, n, pa, pc in candidates:
        print(f"{a:<12s} {c:<12s} {n:>4d} {pa:<12s} {pc:<12s}")

    # 输出可导入 JSON 片段
    print("\n可导入 word_confusion.json 的片段：")
    obj = {a: c for a, c, n, pa, pc in candidates}
    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
