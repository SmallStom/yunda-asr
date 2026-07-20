"""基于 qwen3-asr 真实错误对构建 v3_qwen3 提示词，替换 v2 中的 few-shot 示例.

从 qwen3 全量评估报告中挑选代表性错误样本（排除 100 条测试集），生成 few-shot，
写入 src/prompts/v3_qwen3/system.txt，并更新 registry.json。
原始 v1/v2 不受影响。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
SRC_SAMPLES = ROOT / "data" / "asr_testset" / "qwen3_fewshot_samples.jsonl"
V2_SYSTEM = ROOT / "src" / "prompts" / "v2" / "system.txt"
V3_DIR = ROOT / "src" / "prompts" / "v3_qwen3"
REGISTRY = ROOT / "src" / "prompts" / "registry.json"

# 代表性场景关键词，尽量覆盖不同错误类型
SCENE_KEYWORDS = [
    "销记", "消极", "公务", "工务",
    "无表示", "人物", "仍无",
    "联锁", "连锁", "实验", "试验",
    "进路", "进入",
    "引导", "总锁闭", "计数器",
    "道岔", "扳动", "密贴",
    "凭证", "接近",
    "红光带", "轨道电路",
    "手摇把", "转辙机",
    "出务", "备品",
]


def load_test_ids():
    ids = set()
    with open(TESTSET, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def pick_examples(n=15):
    records = []
    with open(SRC_SAMPLES, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    from tests.utils.metrics import cer
    for r in records:
        r["cer_raw"] = cer(r["asr"], r["correct"])
    details = [r for r in records if r["cer_raw"] > 0.05]

    # 按场景关键词去重挑选，保证多样性
    picked = []
    picked_ids = set()
    seen_scenes = set()
    # 先挑包含关键词的
    for kw in SCENE_KEYWORDS:
        for d in details:
            if d["id"] in picked_ids:
                continue
            if kw in d["asr"] or kw in d["correct"]:
                scene = kw
                if scene in seen_scenes:
                    continue
                picked.append(d)
                picked_ids.add(d["id"])
                seen_scenes.add(scene)
                break
        if len(picked) >= n:
            break
    # 不足则按 cer 降序补
    if len(picked) < n:
        for d in sorted(details, key=lambda x: x.get("cer_raw", 0), reverse=True):
            if d["id"] in picked_ids:
                continue
            picked.append(d)
            picked_ids.add(d["id"])
            if len(picked) >= n:
                break
    return picked[:n]


def build_few_shot_block(examples):
    lines = ["## 六、Few-shot示例（基于 qwen3-asr 真实错误对）", ""]
    for i, d in enumerate(examples, 1):
        lines.append(f"【示例{i}】")
        lines.append(f"输入：{d['asr']}")
        lines.append(f"输出：{d['correct']}")
        lines.append("")
    return "\n".join(lines)


def main():
    examples = pick_examples(15)
    print(f"[INFO] 挑选 {len(examples)} 条 qwen3 真实错误示例")
    for d in examples:
        print(f"  {d['id']}  cer={d.get('cer_raw', 0)}")

    v2_text = V2_SYSTEM.read_text(encoding="utf-8")
    marker = "## 六、Few-shot示例"
    idx = v2_text.find(marker)
    if idx == -1:
        raise RuntimeError("v2 system.txt 找不到 Few-shot 段落")
    head = v2_text[:idx]
    new_block = build_few_shot_block(examples)
    v3_text = head + new_block

    V3_DIR.mkdir(parents=True, exist_ok=True)
    (V3_DIR / "system.txt").write_text(v3_text, encoding="utf-8")
    print(f"[INFO] 写入 {V3_DIR / 'system.txt'}")

    # 更新 registry
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg["versions"]["v3_qwen3"] = {
        "system": "v3_qwen3/system.txt",
        "user_template": "v2/user_template.txt",
        "created_at": "2026-07-06",
        "description": "qwen3-asr 真实错误 few-shot，其余规则同 v2",
    }
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 更新 {REGISTRY}")


if __name__ == "__main__":
    main()
