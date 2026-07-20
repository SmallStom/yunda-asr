"""命令行交互式演示：铁路调度ASR后处理纠错系统.

当Gradio无法启动时使用，提供相同的核心功能演示.

启动：
    python scripts/console_demo.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PostCorrectionPipeline
from src.semantic_refiner import PromptLoader


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_step(step: int, name: str, text: str):
    print(f"\n[Step {step}] {name}:")
    print(f"  {text}")


def save_hotwords(hotwords_text: str) -> None:
    """保存用户编辑的热词."""
    if not hotwords_text.strip():
        return

    hotwords = [w.strip() for w in hotwords_text.replace("，", ",").replace("\n", ",").split(",") if w.strip()]
    lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    hotwords_file = lexicon_dir / "hotwords.json"

    # 备份
    backup_file = lexicon_dir / "hotwords.json.bak"
    if hotwords_file.exists() and not backup_file.exists():
        hotwords_file.rename(backup_file)

    with open(hotwords_file, "w", encoding="utf-8") as f:
        json.dump(hotwords, f, ensure_ascii=False, indent=2)


def restore_hotwords() -> None:
    lexicon_dir = Path(__file__).parent.parent / "data" / "lexicon"
    hotwords_file = lexicon_dir / "hotwords.json"
    backup_file = lexicon_dir / "hotwords.json.bak"

    if backup_file.exists():
        if hotwords_file.exists():
            hotwords_file.unlink()
        backup_file.rename(hotwords_file)


def show_prompts(version: str):
    os.environ["LLM_PROMPT_VERSION"] = version
    system_prompt, user_template = PromptLoader.load()

    print_section(f"当前 Prompt 版本: {version}")
    print("\n[System Prompt 前500字]:")
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    print("\n[User Template]:")
    print(user_template)


def run_demo():
    print_section("铁路调度ASR后处理纠错系统 - 命令行演示")

    # 输入选择
    print("\n请选择输入方式：")
    print("1. 文本输入")
    print("2. 从文件读取ASR文本")
    choice = input("输入选项 (1/2): ").strip()

    if choice == "2":
        file_path = input("请输入文件路径: ").strip()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_text = f.read().strip()
        except Exception as e:
            print(f"读取文件失败: {e}")
            return
    else:
        original_text = input("\n请输入ASR原始文本:\n").strip()

    if not original_text:
        print("输入为空，退出")
        return

    # 配置
    print("\n请选择配置：")
    semantic_mode = input("Layer 4 模式 (baseline/rag/harness，默认rag): ").strip() or "rag"
    prompt_version = input("Prompt版本 (v1/v2，默认v2): ").strip() or "v2"
    hotwords_text = input("热词（逗号分隔，可选）: ").strip()

    # 保存热词
    save_hotwords(hotwords_text)

    try:
        # 设置prompt版本并初始化pipeline
        os.environ["LLM_PROMPT_VERSION"] = prompt_version
        print("\n[INFO] 正在初始化Pipeline...")
        pipeline = PostCorrectionPipeline()
        pipeline.warmup()

        # 运行纠错
        print("[INFO] 正在执行纠错...")
        result = pipeline.run(
            original_text,
            layers=[1, 2, 3],
            enable_semantic=True,
            semantic_mode=semantic_mode,
        )

        # 展示结果
        print_section("分步改写结果")
        print_step(0, "ASR原始输出", original_text)
        print_step(1, "Layer 1 文本预处理", result.layer_outputs.get("layer1", original_text))
        print_step(2, "Layer 2 词典纠错", result.layer_outputs.get("layer2", ""))
        print_step(3, "Layer 3 上下文纠错", result.layer_outputs.get("layer3", ""))
        print_step(4, f"Layer 4 语义精修 ({semantic_mode} + {prompt_version})", result.layer_outputs.get("layer4", ""))

        print_section("最终输出")
        print(f"  {result.corrected}")

        # 展示prompt
        show_prompt = input("\n是否查看当前Prompt? (y/n): ").strip().lower()
        if show_prompt in ("y", "yes", "是"):
            show_prompts(prompt_version)

    finally:
        restore_hotwords()
        print("\n[INFO] 热词已恢复")


def main():
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        restore_hotwords()


if __name__ == "__main__":
    main()
