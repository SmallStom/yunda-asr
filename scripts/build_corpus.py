"""语料合并与清洗脚本."""

import json
import re
from pathlib import Path


def parse_text_file(filepath: Path) -> list[str]:
    """解析标注文本文件，提取纯文本内容."""
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: train_XXXX 文本内容
            # 分割第一个空格
            parts = line.split(" ", 1)
            if len(parts) == 2:
                texts.append(parts[1].strip())
            else:
                # 可能没有空格分隔的情况
                texts.append(line.strip())
    return texts


def clean_text(text: str) -> str:
    """清洗单条文本."""
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)
    # 去除首尾空白
    text = text.strip()
    return text


def augment_corpus(texts: list[str], num_variants: int = 10) -> list[str]:
    """通过同义替换扩增语料."""
    # 定义可替换的实体池
    stations = ["A站", "B站", "C站", "D站", "甲站", "乙站"]
    tracks = ["1道", "2道", "3道", "I道", "II道", "4道", "5道"]
    switches = ["6号", "8号", "10号", "12号", "13号", "14号", "16号", "18号", "24号"]
    trains = ["48015次", "48017次", "48613次", "48615次", "G534次", "G541次", "G542次", "G543次", "G544次", "57001次", "57002次"]
    
    import random
    random.seed(42)
    
    augmented = []
    for text in texts:
        augmented.append(text)
        # 为每条文本生成num_variants个变体
        for _ in range(num_variants):
            variant = text
            # 替换站名
            if "某站" in variant:
                variant = variant.replace("某站", random.choice(stations))
            # 随机替换车次（保持原格式）
            for train in trains:
                if train in variant:
                    variant = variant.replace(train, random.choice(trains), 1)
                    break  # 只替换一次，避免过度替换
            # 随机替换道岔号
            for switch in switches:
                if switch + "道岔" in variant or switch + "号道岔" in variant:
                    new_switch = random.choice(switches)
                    variant = variant.replace(switch + "号道岔", new_switch + "号道岔", 1)
                    variant = variant.replace(switch + "道岔", new_switch + "号道岔", 1)
                    break
            # 随机替换股道
            for track in tracks:
                if track in variant:
                    variant = variant.replace(track, random.choice(tracks), 1)
                    break
            if variant != text:
                augmented.append(variant)
    
    return augmented


def main():
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset" / "语音识别样本集"
    corpus_dir = project_root / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    all_texts = []
    
    # 1. 四川话标注语料
    sichuan_file = dataset_dir / "录音四川话-长文本" / "表格.txt"
    if sichuan_file.exists():
        texts = parse_text_file(sichuan_file)
        all_texts.extend(texts)
        print(f"[build_corpus] 加载四川话语料: {len(texts)} 条")
    
    # 2. 普通话标注语料
    for subdir in ["2", "3"]:
        text_file = dataset_dir / "录音普通话-长文本" / subdir / "text.txt"
        if text_file.exists():
            texts = parse_text_file(text_file)
            all_texts.extend(texts)
            print(f"[build_corpus] 加载普通话语料 ({subdir}): {len(texts)} 条")
    
    # 清洗
    all_texts = [clean_text(t) for t in all_texts]
    all_texts = list(dict.fromkeys(all_texts))  # 去重
    print(f"[build_corpus] 去重后原始语料: {len(all_texts)} 条")
    
    # 保存原始语料
    raw_corpus_file = corpus_dir / "railway_corpus_raw.txt"
    with open(raw_corpus_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")
    print(f"[build_corpus] 原始语料已保存: {raw_corpus_file}")
    
    # 扩增
    augmented = augment_corpus(all_texts, num_variants=15)
    augmented = list(dict.fromkeys(augmented))
    print(f"[build_corpus] 扩增后语料: {len(augmented)} 条")
    
    # 保存扩增语料
    corpus_file = corpus_dir / "railway_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in augmented:
            f.write(text + "\n")
    print(f"[build_corpus] 扩增语料已保存: {corpus_file}")
    
    # 构建ASR错误平行语对（规则模拟）
    error_pairs = []
    for text in all_texts[:100]:  # 先用前100条生成模拟错误对
        # 模拟几种常见错误
        # 1. 同音替换
        err1 = text.replace("道岔", "道差").replace("信号", "新号")
        if err1 != text:
            error_pairs.append({"asr": err1, "correct": text, "error_types": ["phonetic"]})
        
        # 2. 数字口语化
        err2 = re.sub(r'(\d+)号', lambda m: cn2an_transform(m.group(1)) + '号', text)
        if err2 != text:
            error_pairs.append({"asr": err2, "correct": text, "error_types": ["number"]})
    
    pairs_file = corpus_dir / "asr_error_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for pair in error_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[build_corpus] 模拟错误对已保存: {pairs_file} ({len(error_pairs)} 对)")


def cn2an_transform(num_str: str) -> str:
    """将阿拉伯数字转为中文数字（简单版）."""
    mapping = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    return ''.join(mapping.get(c, c) for c in num_str)


if __name__ == "__main__":
    main()
