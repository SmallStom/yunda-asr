"""清理 aliases.json 中的危险语义归一化映射.

删除会引入错误的映射：
- "进站信号机"→"地面信号机"（不应归一化，进站/出站是具体类型）
- "出站信号机"→"地面信号机"
- "通过信号机"→"地面信号机"
- "色灯信号机"→"地面信号机"
- "信号灯"→"地面信号机"
"""

import json
from pathlib import Path

# 需要删除的危险映射（会将正确的具体术语归一化为大类，导致信息丢失）
DANGEROUS_MAPPINGS = {
    "进站信号机", "出站信号机", "通过信号机", "色灯信号机", "信号灯",
    "进站", "出站",  # 这些会被映射为信号机类型，但在后处理阶段不应做
}

def main():
    alias_file = Path("data/lexicon/aliases.json")

    with open(alias_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"原始aliases: {len(raw)} 条")

    removed = {}
    for key in list(raw.keys()):
        if key in DANGEROUS_MAPPINGS:
            removed[key] = raw.pop(key)

    print(f"删除危险映射: {len(removed)} 条")
    for k, v in removed.items():
        print(f"  {k} → {v}")

    with open(alias_file, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"\n清理后aliases: {len(raw)} 条")


if __name__ == "__main__":
    main()
