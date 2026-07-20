"""端到端测试.

基于真实ASR样本集评估各层纠错效果.
"""

import json
import time
from pathlib import Path

import pytest

from src.pipeline import PostCorrectionPipeline
from tests.utils.metrics import cer, entity_fidelity, load_asr_test_pairs, term_accuracy


# 全局流水线实例（避免重复初始化）
_pipeline = None


def get_pipeline() -> PostCorrectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PostCorrectionPipeline()
        _pipeline.warmup()
    return _pipeline


# 加载测试样本（全局只加载一次）
def _get_test_samples(limit: int = 50) -> list:
    """加载并筛选有效测试样本."""
    records = load_asr_test_pairs()
    # 按子目录分散采样，确保覆盖不同场景
    samples = []
    subdirs = {"1": [], "2": [], "3": [], "4": []}
    for r in records:
        sid = r.get("id", "")
        for d in subdirs:
            if sid.startswith(d + "/"):
                subdirs[d].append(r)
                break

    # 每个子目录均匀采样
    per_dir = max(1, limit // len(subdirs))
    for d, lst in subdirs.items():
        samples.extend(lst[:per_dir])

    # 如果不够，补充
    if len(samples) < limit:
        seen = {s["id"] for s in samples}
        for r in records:
            if r["id"] not in seen:
                samples.append(r)
            if len(samples) >= limit:
                break

    return samples[:limit]


# 延迟加载，避免pytest收集阶段耗时
_TEST_SAMPLES = None


def get_test_samples() -> list:
    global _TEST_SAMPLES
    if _TEST_SAMPLES is None:
        _TEST_SAMPLES = _get_test_samples(limit=50)
    return _TEST_SAMPLES


@pytest.fixture(scope="module")
def pipeline():
    return get_pipeline()


class TestEndToEnd:
    """端到端测试类."""

    @pytest.mark.parametrize("record", get_test_samples(), ids=lambda r: r.get("id", "unknown"))
    def test_single_sample(self, pipeline, record):
        """对单个ASR样本运行纠错流水线，验证不抛异常且输出合理."""
        asr_text = record["asr"]
        correct_text = record["correct"]

        result = pipeline.run(asr_text, layers=[1, 2, 3])

        # 基本断言：输出非空
        assert result.corrected is not None
        # 输出不应为空（除非输入为空）
        if asr_text.strip():
            assert result.corrected.strip() != ""

        # 计算CER
        cer_before = cer(asr_text, correct_text)
        cer_after = cer(result.corrected, correct_text)

        # 记录到pytest的node属性中（供汇总测试使用）
        # 注意：pytest不会在parametrize测试中自动共享数据，
        # 因此汇总测试会重新运行所有样本
        print(f"\n[{record['id']}] CER: {cer_before:.3f} -> {cer_after:.3f}")

    def test_overall_metrics(self, pipeline):
        """汇总评估指标并生成报告."""
        samples = get_test_samples()
        if not samples:
            pytest.skip("无有效测试样本")

        results = []
        total_cer_before = 0.0
        total_cer_after = 0.0
        total_term_acc_before = 0.0
        total_term_acc_after = 0.0
        total_entity_fid_before = 0.0
        total_entity_fid_after = 0.0
        improved_count = 0
        unchanged_count = 0
        degraded_count = 0

        layer_stats = {
            "preprocessor": {"triggered": 0, "samples": 0},
            "dictionary": {"triggered": 0, "samples": 0},
            "context": {"triggered": 0, "samples": 0},
            "semantic": {"triggered": 0, "samples": 0},
        }

        for record in samples:
            asr_text = record["asr"]
            correct_text = record["correct"]

            t0 = time.time()
            result = pipeline.run(asr_text, layers=[1, 2, 3])
            latency = time.time() - t0

            cer_before = cer(asr_text, correct_text)
            cer_after = cer(result.corrected, correct_text)

            term_acc_before, _, _ = term_accuracy(asr_text, correct_text)
            term_acc_after, term_hits, term_total = term_accuracy(result.corrected, correct_text)

            entity_fid_before, _, _ = entity_fidelity(asr_text, correct_text)
            entity_fid_after, entity_hits, entity_total = entity_fidelity(result.corrected, correct_text)

            total_cer_before += cer_before
            total_cer_after += cer_after
            total_term_acc_before += term_acc_before
            total_term_acc_after += term_acc_after
            total_entity_fid_before += entity_fid_before
            total_entity_fid_after += entity_fid_after

            if cer_after < cer_before - 0.001:
                improved_count += 1
            elif cer_after > cer_before + 0.001:
                degraded_count += 1
            else:
                unchanged_count += 1

            for layer in result.layers_applied:
                if layer in layer_stats:
                    layer_stats[layer]["triggered"] += 1
            for layer in layer_stats:
                layer_stats[layer]["samples"] += 1

            results.append({
                "id": record["id"],
                "asr": asr_text,
                "correct": correct_text,
                "corrected": result.corrected,
                "cer_before": round(cer_before, 4),
                "cer_after": round(cer_after, 4),
                "term_hits": term_hits,
                "term_total": term_total,
                "entity_hits": entity_hits,
                "entity_total": entity_total,
                "layers_applied": result.layers_applied,
                "latency_ms": round(latency * 1000, 2),
            })

        n = len(samples)
        avg_cer_before = total_cer_before / n
        avg_cer_after = total_cer_after / n
        avg_term_before = total_term_acc_before / n
        avg_term_after = total_term_acc_after / n
        avg_entity_before = total_entity_fid_before / n
        avg_entity_after = total_entity_fid_after / n

        report = {
            "summary": {
                "total_samples": n,
                "avg_cer_before": round(avg_cer_before, 4),
                "avg_cer_after": round(avg_cer_after, 4),
                "cer_improvement": round(avg_cer_before - avg_cer_after, 4),
                "avg_term_accuracy_before": round(avg_term_before, 4),
                "avg_term_accuracy_after": round(avg_term_after, 4),
                "avg_entity_fidelity_before": round(avg_entity_before, 4),
                "avg_entity_fidelity_after": round(avg_entity_after, 4),
                "improved_count": improved_count,
                "unchanged_count": unchanged_count,
                "degraded_count": degraded_count,
            },
            "layer_stats": {
                layer: {
                    "triggered": stats["triggered"],
                    "trigger_rate": round(stats["triggered"] / max(stats["samples"], 1), 4),
                }
                for layer, stats in layer_stats.items()
            },
            "worst_cases": sorted(
                [r for r in results if r["cer_after"] > r["cer_before"]],
                key=lambda x: x["cer_after"] - x["cer_before"],
                reverse=True,
            )[:10],
            "best_cases": sorted(
                [r for r in results if r["cer_after"] < r["cer_before"]],
                key=lambda x: x["cer_before"] - x["cer_after"],
                reverse=True,
            )[:10],
            "details": results,
        }

        # 保存报告
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / f"e2e_report_{int(time.time())}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[E2E] 评估报告已保存: {report_path}")

        # 断言：整体CER应不劣化（允许微小波动）
        assert avg_cer_after <= avg_cer_before + 0.05, (
            f"整体CER劣化过多: {avg_cer_before:.4f} -> {avg_cer_after:.4f}"
        )
        # 断言：至少有一定比例的样本得到改善或保持不变
        assert improved_count + unchanged_count >= n * 0.7, (
            f"改善+保持不变比例过低: {(improved_count + unchanged_count) / n:.2%}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
