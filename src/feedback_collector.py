"""错误案例收集与分类模块.

收集纠错失败案例，自动分析根因（定位到具体层级），
并持久化到 data/feedback/ 目录，供后续词典维护使用.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.pipeline import PipelineResult


@dataclass
class CorrectionFailure:
    timestamp: str
    original: str           # ASR原始输出
    corrected: str          # 系统纠错结果
    expected: str           # 人工标注的正确文本
    failure_type: str       # "漏纠" | "过纠" | "数字篡改" | "术语错误" | "其他"
    layer_responsible: str  # "layer1" | "layer2" | "layer3" | "layer4" | "unknown"
    confidence: float       # 系统置信度（预留，当前固定为0.0）
    details: dict           # 额外详情（如各层输出）


class FeedbackCollector:
    """纠错失败案例收集器."""

    FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"

    def __init__(self):
        self.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        original: str,
        corrected: str,
        expected: str,
        pipeline_result: PipelineResult,
    ) -> CorrectionFailure:
        """收集一个失败案例并分析根因."""
        failure_type = self._classify_failure(original, corrected, expected)
        layer_responsible = self._analyze_root_cause(
            original, corrected, expected, pipeline_result
        )

        failure = CorrectionFailure(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            original=original,
            corrected=corrected,
            expected=expected,
            failure_type=failure_type,
            layer_responsible=layer_responsible,
            confidence=0.0,
            details={
                "layer_outputs": pipeline_result.layer_outputs,
                "layers_applied": pipeline_result.layers_applied,
            },
        )

        self._save(failure)
        return failure

    def _classify_failure(self, original: str, corrected: str, expected: str) -> str:
        """分类失败类型."""
        # 数字篡改检测
        orig_numbers = set(re.findall(r"\d+", original))
        corr_numbers = set(re.findall(r"\d+", corrected))
        exp_numbers = set(re.findall(r"\d+", expected))
        if corr_numbers != exp_numbers and orig_numbers == exp_numbers:
            return "数字篡改"

        # 术语错误：原文有正确术语但被改错
        # 简单判断：如果expected中的术语在corrected中缺失
        if corrected == original:
            return "漏纠"

        # 如果corrected比original离expected更远 → 过纠
        # 这里用简单字符串包含判断
        if original != corrected and corrected != expected:
            return "过纠"

        return "其他"

    def _analyze_root_cause(
        self,
        original: str,
        corrected: str,
        expected: str,
        result: PipelineResult,
    ) -> str:
        """分析根因，定位到首次出现偏差的层级.

        策略：逐层检查，找到第一层"使输出偏离 expected 更远"的层级。
        若所有层都未使差异扩大，则定位为最后一层输出仍不等于 expected 的层级。
        """
        outputs = result.layer_outputs
        layers = result.layers_applied

        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        prev = original
        # 逐层检查：找到第一层使差异扩大的
        from tests.utils.metrics import cer as _cer

        prev_cer = _cer(prev, expected)

        for layer_name in layer_names:
            current = outputs.get(layer_name, prev)
            if current == prev:
                # 该层未做修改，跳过
                prev = current
                continue

            # 该层有修改，检查修改后是否与 expected 差异更大
            current_cer = _cer(current, expected)
            if current_cer > prev_cer + 0.01:
                # 修改使差异显著扩大，该层负责
                return layer_name
            prev = current
            prev_cer = current_cer

        # 所有层都未使差异扩大，但仍不等于 expected
        # 定位为最后一层输出不等于 expected 的层级
        prev = original
        last_bad_layer = "unknown"
        for layer_name in layer_names:
            current = outputs.get(layer_name, prev)
            if current != expected:
                last_bad_layer = layer_name
            prev = current

        return last_bad_layer

    def _save(self, failure: CorrectionFailure) -> None:
        """保存失败案例到JSONL文件."""
        date_str = time.strftime("%Y%m%d")
        file_path = self.FEEDBACK_DIR / f"failures_{date_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(failure), ensure_ascii=False) + "\n")

    def load_failures(self, days: int = 7) -> List[CorrectionFailure]:
        """加载最近N天的失败案例."""
        failures = []
        for i in range(days):
            date_str = time.strftime("%Y%m%d", time.localtime(time.time() - i * 86400))
            file_path = self.FEEDBACK_DIR / f"failures_{date_str}.jsonl"
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        failures.append(CorrectionFailure(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return failures

    def generate_weekly_report(self) -> Path:
        """生成周汇总报告."""
        failures = self.load_failures(days=7)

        stats = {
            "total": len(failures),
            "by_type": {},
            "by_layer": {},
        }

        for f in failures:
            stats["by_type"][f.failure_type] = stats["by_type"].get(f.failure_type, 0) + 1
            stats["by_layer"][f.layer_responsible] = stats["by_layer"].get(f.layer_responsible, 0) + 1

        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "period_days": 7,
            "stats": stats,
            "samples": [asdict(f) for f in failures[:20]],  # 保留前20条示例
        }

        report_path = self.FEEDBACK_DIR / "weekly_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report_path


def collect_failure(
    original: str,
    corrected: str,
    expected: str,
    pipeline_result: PipelineResult,
) -> CorrectionFailure:
    """快捷函数：收集单个失败案例."""
    collector = FeedbackCollector()
    return collector.collect(original, corrected, expected, pipeline_result)
