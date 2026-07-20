"""基于 v2 创建 v4_qwen3_opt 优化提示词：保留 v2 全部规则，补充 qwen3-asr 常见误识别映射.

仅新增一节 qwen3 常见错误补充，不改 v2 原文，通过环境变量 LLM_PROMPT_VERSION=v4_qwen3_opt 切换。
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
V2_SYSTEM = ROOT / "src" / "prompts" / "v2" / "system.txt"
V4_DIR = ROOT / "src" / "prompts" / "v4_qwen3_opt"
REGISTRY = ROOT / "src" / "prompts" / "registry.json"

QWEN3_SUPPLEMENT = """## 七、qwen3-asr 常见误识别补充（模型特有错误）

以下错误在 qwen3-asr 输出中高频出现，按语境修正：

| ASR输出 | 正确 | 触发语境 |
|---|---|---|
| 消极 | 销记 | 工务/电务后 |
| 人物表示 | 仍无表示 | 控制台/道岔后 |
| 专车机/转车机 | 转辙机 | 手摇把/钥匙/道岔语境 |
| 战线部/站线部 | 占线簿 | 填写后 |
| 家岗 | 加岗 | 人员/撤回语境 |
| 公务 | 工务 | 设备销记语境 |
| 消记/消起 | 销记 | 工务/电务后 |
| 连锁 | 联锁 | 信号/试验语境 |
| 实验 | 试验 | 设备测试语境 |
| 进入 | 进路 | 发车/接车/确认后（非地点） |

**数字规则强化**：qwen3 常输出中文数字（十八号、零零一），必须转为阿拉伯数字（18号、001）。

**标点规则强化**：原ASR无标点时不要补句号；仅在原ASR已有标点时保留对应断句。不要新增逗号、句号。

---

"""


def main():
    v2_text = V2_SYSTEM.read_text(encoding="utf-8")
    marker = "## 六、Few-shot示例"
    idx = v2_text.find(marker)
    if idx == -1:
        raise RuntimeError("v2 system.txt 找不到 Few-shot 段落")
    head = v2_text[:idx]
    tail = v2_text[idx:]
    v4_text = head + QWEN3_SUPPLEMENT + tail

    V4_DIR.mkdir(parents=True, exist_ok=True)
    (V4_DIR / "system.txt").write_text(v4_text, encoding="utf-8")
    print(f"[INFO] 写入 {V4_DIR / 'system.txt'}")

    import json
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg["versions"]["v4_qwen3_opt"] = {
        "system": "v4_qwen3_opt/system.txt",
        "user_template": "v2/user_template.txt",
        "created_at": "2026-07-07",
        "description": "v2 + qwen3常见误识别补充 + 标点/数字规则强化",
    }
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 更新 {REGISTRY}")


if __name__ == "__main__":
    main()
