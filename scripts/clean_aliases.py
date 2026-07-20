"""清理 aliases.json：删除自映射和语义归一化，只保留同音/近音误识别."""

import json
from pathlib import Path

def is_phonetic_similarity(alias: str, canonical: str) -> bool:
    """判断alias是否是canonical的近音误识别（而非语义归一化）.

    判断标准：
    1. 长度相近（差值<=2）
    2. 不是完全相同（自映射）
    3. 不是明显的语义归一化（如"进站"→"进站信号机"）
    """
    if alias == canonical:
        return False  # 自映射

    len_diff = abs(len(alias) - len(canonical))
    if len_diff > 2:
        return False  # 长度差太大，可能是语义归一化

    # 如果canonical是alias的子串或反过来，且长度差>1，可能是语义归一化
    if len_diff > 1:
        if alias in canonical or canonical in alias:
            return False

    return True


def main():
    lexicon_dir = Path("data/lexicon")
    alias_file = lexicon_dir / "aliases.json"

    with open(alias_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"原始aliases: {len(raw)} 条")

    # 统计
    self_mapped = 0
    semantic_normalized = 0
    phonetic_kept = 0

    cleaned = {}
    for alias, canonical in raw.items():
        if alias == canonical:
            self_mapped += 1
            continue

        if not is_phonetic_similarity(alias, canonical):
            semantic_normalized += 1
            continue

        cleaned[alias] = canonical
        phonetic_kept += 1

    print(f"自映射（删除）: {self_mapped} 条")
    print(f"语义归一化（删除）: {semantic_normalized} 条")
    print(f"近音误识别（保留）: {phonetic_kept} 条")

    # 保存清理后的文件
    backup_path = lexicon_dir / "backup_pre_refactor" / "aliases.json"
    if backup_path.exists():
        with open(alias_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print(f"\n清理后aliases已保存: {alias_file} ({len(cleaned)} 条)")
    else:
        print("[ERROR] 备份文件不存在，未保存")


if __name__ == "__main__":
    main()
