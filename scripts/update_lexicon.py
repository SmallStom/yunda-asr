"""词典增量更新脚本.

从反馈数据中提取候选别名更新，生成待审核CSV，支持导入审核结果.

Usage:
    python scripts/update_lexicon.py --review --output pending_aliases.csv
    python scripts/update_lexicon.py --import approved_aliases.csv
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback_collector import FeedbackCollector


LEXICON_DIR = Path("data/lexicon")
ALIASES_FILE = LEXICON_DIR / "aliases.json"
TERMS_FILE = LEXICON_DIR / "railway_terms.json"
BACKUP_DIR = LEXICON_DIR / "backups"


def parse_args():
    parser = argparse.ArgumentParser(description="词典增量更新")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--review",
        action="store_true",
        help="生成待审核的别名列表",
    )
    group.add_argument(
        "--import-csv",
        type=Path,
        metavar="CSV",
        help="导入人工审核后的CSV文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pending_aliases.csv"),
        help="待审核列表输出路径 (默认: pending_aliases.csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="读取最近N天的反馈 (默认: 7)",
    )
    return parser.parse_args()


def load_aliases() -> Dict[str, str]:
    """加载当前别名映射."""
    if not ALIASES_FILE.exists():
        return {}
    with open(ALIASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_aliases(aliases: Dict[str, str]) -> None:
    """保存别名映射并备份旧版本."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if ALIASES_FILE.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"aliases_{timestamp}.json"
        shutil.copy2(ALIASES_FILE, backup_path)
        print(f"[INFO] 旧词典已备份: {backup_path}")

    with open(ALIASES_FILE, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 别名映射已更新: {ALIASES_FILE}")


def extract_candidates(failures) -> List[dict]:
    """从失败案例中提取候选别名更新."""
    candidates = []
    seen: Set[str] = set()

    for f in failures:
        original = f.original
        expected = f.expected

        # 漏纠：original中有错误别名，expected中是正确术语
        if f.failure_type == "漏纠":
            # 简单启发式：找出expected中有但original中没有的术语
            # 这里用模糊匹配，实际生产中可能需要更复杂的对齐算法
            cand = _extract_alias_pair(original, expected)
            if cand and cand["alias"] not in seen:
                seen.add(cand["alias"])
                candidates.append(cand)

        # 过纠：需要删除或修正的别名
        elif f.failure_type == "过纠":
            cand = _extract_overcorrection_pair(original, corrected=f.corrected, expected=expected)
            if cand and cand["alias"] not in seen:
                seen.add(cand["alias"])
                candidates.append(cand)

    return candidates


def _extract_alias_pair(original: str, expected: str) -> Optional[dict]:
    """从漏纠案例中提取别名-标准词对（简化启发式）."""
    # 启发式：找出expected中和original长度相近但不同的子串
    # 这是一个简化版，实际应使用编辑距离对齐
    if len(original) < 2 or len(expected) < 2:
        return None

    # 简单尝试：如果expected包含一个2-4字的词，而original包含一个音似替代
    # 这里仅做演示，返回一个启发式候选
    # 实际生产中应使用pypinyin对比找出音似差异词
    import difflib
    sm = difflib.SequenceMatcher(None, original, expected)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) <= 4 and (j2 - j1) <= 4:
            alias = original[i1:i2]
            canonical = expected[j1:j2]
            if alias and canonical and alias != canonical:
                return {
                    "operation": "add",
                    "alias": alias,
                    "canonical": canonical,
                    "source": f"original: {original[:30]}...",
                    "confidence": 0.5,
                }
    return None


def _extract_overcorrection_pair(original: str, corrected: str, expected: str) -> Optional[dict]:
    """从过纠案例中提取需要删除/修正的别名对."""
    import difflib
    sm = difflib.SequenceMatcher(None, corrected, expected)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) <= 4:
            wrong = corrected[i1:i2]
            right = expected[j1:j2]
            if wrong and right:
                return {
                    "operation": "review",
                    "alias": wrong,
                    "canonical": right,
                    "source": f"corrected: {corrected[:30]}...",
                    "confidence": 0.3,
                }
    return None


def generate_review_csv(candidates: List[dict], output_path: Path) -> None:
    """生成待审核CSV文件."""
    fieldnames = ["operation", "alias", "canonical", "source", "confidence", "approved"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            row = dict(cand)
            row["approved"] = ""  # 人工填写 yes/no
            writer.writerow(row)
    print(f"[INFO] 待审核列表已生成: {output_path} ({len(candidates)} 条)")


def import_csv(csv_path: Path) -> None:
    """导入审核后的CSV，更新词典."""
    aliases = load_aliases()
    added = 0
    removed = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            approved = row.get("approved", "").strip().lower()
            if approved not in ("yes", "y", "true", "1"):
                continue

            operation = row.get("operation", "add")
            alias = row.get("alias", "").strip()
            canonical = row.get("canonical", "").strip()

            if not alias or not canonical:
                continue

            if operation == "add":
                if alias in aliases and aliases[alias] != canonical:
                    print(f"[WARN] 别名冲突: {alias} -> {aliases[alias]} (现有) vs {canonical} (新)")
                aliases[alias] = canonical
                added += 1
            elif operation == "remove":
                if alias in aliases:
                    del aliases[alias]
                    removed += 1

    if added or removed:
        save_aliases(aliases)
        print(f"[INFO] 更新完成: 新增 {added} 条, 删除 {removed} 条")
    else:
        print("[INFO] 无变更")


def main():
    args = parse_args()

    if args.review:
        print(f"[INFO] 加载最近 {args.days} 天的反馈数据...")
        collector = FeedbackCollector()
        failures = collector.load_failures(days=args.days)
        print(f"[INFO] 加载失败案例: {len(failures)} 条")

        candidates = extract_candidates(failures)
        print(f"[INFO] 提取候选更新: {len(candidates)} 条")

        if candidates:
            generate_review_csv(candidates, args.output)
            print("[INFO] 请人工审核CSV后，执行:")
            print(f"  python scripts/update_lexicon.py --import-csv {args.output}")
        else:
            print("[INFO] 未发现新的候选别名")

    elif args.import_csv:
        if not args.import_csv.exists():
            print(f"[ERROR] 文件不存在: {args.import_csv}")
            sys.exit(1)
        import_csv(args.import_csv)


if __name__ == "__main__":
    main()
