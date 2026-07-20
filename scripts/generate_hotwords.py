"""生成ASR热词列表.

从专有名词库提取铁路领域术语，生成VibeVoice-ASR热词文件.
热词通过提升模型对领域专有名词的识别准确率，从源头改善ASR质量.
"""

import json
from pathlib import Path
from typing import List, Set


def generate_hotwords(lexicon_dir: Path | str | None = None) -> List[str]:
    """从术语库生成热词列表.

    策略：
        1. 从 railway_terms.json 提取所有标准术语(canonical)
        2. 从 aliases.json 提取所有标准词(值)
        3. 过滤掉过短(<2字)或过长(>10字)的词
        4. 去重并按字数排序

    Returns:
        热词列表
    """
    if lexicon_dir is None:
        lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    else:
        lexicon_dir = Path(lexicon_dir)

    hotwords_set: Set[str] = set()

    # 1. 从 railway_terms.json 提取标准术语
    terms_file = lexicon_dir / "railway_terms.json"
    if terms_file.exists():
        with open(terms_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for term_info in data.get("terms", []):
                canonical = term_info.get("canonical", "").strip()
                if canonical:
                    hotwords_set.add(canonical)

    # 2. 从 aliases.json 提取标准词（值）
    alias_file = lexicon_dir / "aliases.json"
    if alias_file.exists():
        with open(alias_file, "r", encoding="utf-8") as f:
            aliases = json.load(f)
            for canonical in aliases.values():
                canonical = canonical.strip()
                if canonical:
                    hotwords_set.add(canonical)

    # 3. 过滤：保留2-10字的词，去除含特殊字符的词
    filtered = set()
    for word in hotwords_set:
        # 长度过滤
        if len(word) < 2 or len(word) > 10:
            continue
        # 去除含特殊字符的词（只保留中文、字母、数字）
        if not all(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in word):
            continue
        # 去除纯数字
        if word.isdigit():
            continue
        filtered.add(word)

    # 4. 按字数降序排序（长词优先，有助于ASR匹配）
    hotwords = sorted(filtered, key=lambda x: (-len(x), x))
    return hotwords


def save_hotwords(hotwords: List[str], output_path: Path | str) -> None:
    """保存热词列表到文件.

    生成两种格式：
        - .txt: 每行一个热词（供人查看）
        - .json: JSON数组格式（供程序读取）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TXT格式
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for word in hotwords:
            f.write(word + "\n")

    # JSON格式
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hotwords, f, ensure_ascii=False, indent=2)

    # 逗号分隔格式（VibeVoice-ASR hotwords参数直接使用）
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(hotwords))

    print(f"[INFO] 热词数量: {len(hotwords)}")
    print(f"[INFO] TXT格式: {txt_path}")
    print(f"[INFO] JSON格式: {json_path}")
    print(f"[INFO] CSV格式(逗号分隔): {csv_path}")


def load_hotwords(path: Path | str | None = None) -> List[str]:
    """加载热词列表."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "lexicon" / "hotwords.json"
    else:
        path = Path(path)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".json":
            return json.load(f)
        else:
            return [line.strip() for line in f if line.strip()]


def main():
    """生成并保存热词列表."""
    lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    output_path = lexicon_dir / "hotwords"

    hotwords = generate_hotwords(lexicon_dir)
    save_hotwords(hotwords, output_path)

    # 打印前20个热词作为预览
    print("\n[预览] 前20个热词:")
    for i, word in enumerate(hotwords[:20], 1):
        print(f"  {i:2d}. {word}")


if __name__ == "__main__":
    main()
