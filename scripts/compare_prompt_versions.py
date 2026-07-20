"""对比两个Prompt版本在baseline/rag/harness模式下的效果.

用法:
    python scripts/compare_prompt_versions.py --testset data/asr_testset/asr_test_pairs_elevenlabs.jsonl

输出:
    reports/prompt_version_compare.xlsx
    - Sheet1: 汇总指标对比
    - Sheet2: 逐条样本v1/v2文本对比
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.metrics import cer, is_valid_asr_text


def load_records(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_valid_asr_text(record.get("asr", "")):
                records.append(record)
    return records


def run_eval(testset: Path, version: str, output: Path) -> dict:
    """以指定prompt版本运行evaluate_three_directions.py，返回汇总指标."""
    env = os.environ.copy()
    env["LLM_PROMPT_VERSION"] = version

    cmd = [
        sys.executable,
        "scripts/evaluate_three_directions.py",
        "--testset", str(testset),
    ]

    print(f"\n[INFO] 开始评估 prompt version={version}")
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    print(f"[INFO] version={version} 耗时 {elapsed:.1f}s")

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"评估失败: {version}")

    # 从输出中解析汇总指标
    metrics = {}
    for line in result.stdout.splitlines():
        if version == "v1" and "baseline" in line and "." in line:
            # 首次出现汇总表
            pass
        # 简单解析最后一行汇总
        if line.startswith("模式") or line.startswith("---") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0] in ("baseline", "rag", "harness"):
            mode = parts[0]
            metrics[mode] = {
                "raw_cer": float(parts[1]),
                "rule_cer": float(parts[2]),
                "llm_cer": float(parts[3]),
                "llm_gain": float(parts[4].replace("+", "")),
                "improved": int(parts[5]),
                "degraded": int(parts[6]),
                "unchanged": int(parts[7]),
                "avg_time_ms": int(parts[8].replace("ms", "")),
            }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="对比Prompt版本效果")
    parser.add_argument(
        "--testset",
        type=str,
        default="data/asr_testset/asr_test_pairs_elevenlabs.jsonl",
        help="测试集路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/prompt_version_compare.xlsx",
        help="输出Excel路径",
    )
    args = parser.parse_args()

    testset = Path(args.testset)
    output = Path(args.output)

    # 分别跑v1和v2
    v1_metrics = run_eval(testset, "v1", output)
    v2_metrics = run_eval(testset, "v2", output)

    print("\n" + "=" * 80)
    print("Prompt版本对比汇总")
    print("=" * 80)
    print(f"{'模式':<10} {'版本':<5} {'原始CER':<10} {'规则CER':<10} {'LLM CER':<10} {'改善':<6} {'劣化':<6} {'不变':<6}")
    print("-" * 80)
    for mode in ["baseline", "rag", "harness"]:
        m1 = v1_metrics.get(mode, {})
        m2 = v2_metrics.get(mode, {})
        print(f"{mode:<10} v1    {m1.get('raw_cer', 0):<10.4f} {m1.get('rule_cer', 0):<10.4f} {m1.get('llm_cer', 0):<10.4f} {m1.get('improved', 0):<6} {m1.get('degraded', 0):<6} {m1.get('unchanged', 0):<6}")
        print(f"{mode:<10} v2    {m2.get('raw_cer', 0):<10.4f} {m2.get('rule_cer', 0):<10.4f} {m2.get('llm_cer', 0):<10.4f} {m2.get('improved', 0):<6} {m2.get('degraded', 0):<6} {m2.get('unchanged', 0):<6}")
        delta = m2.get('llm_cer', 0) - m1.get('llm_cer', 0)
        print(f"{'':<10} {'delta':<5} {'':<10} {'':<10} {delta:<+10.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
