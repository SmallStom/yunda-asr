"""LLM纠错对比测试.

使用10条规则纠错后仍错误的hard case，对比四种模式：
1. baseline - 基线LLM纠错（原始SemanticRefiner）
2. nbest - 方向一：N-best候选融合+约束解码
3. rag - 方向二：RAG增强+动态Few-shot
4. fusion - 融合模式：RAG知识增强 + N-best约束解码
"""

import json
import time
from pathlib import Path

import pytest

from src.pipeline import PostCorrectionPipeline
from tests.utils.metrics import cer, entity_fidelity, term_accuracy


# 10条改错后仍然错误的hard case（与test_llm_correction相同）
HARD_CASES = [
    {
        "id": "train_0011",
        "asr": "公修机线路设备正常，电务登记十三号到车，无标识。轨道电路故障，不影响非正常情况下，计划列车及调车作业。",
        "correct": "B站工务销记线路设备正常，电务登记13号道岔无表示属轨道电路故障，不影响非正常情况下的接发列车及调车作业",
    },
    {
        "id": "train_0001_v2",
        "asr": "年前四八六幺五四四到停车开放信吗？",
        "correct": "内勤，48615次，3道停车，开放信号",
    },
    {
        "id": "train_0001_v3",
        "asr": "列车已驶离，到停车开放信号。",
        "correct": "内勤，48615次，3道停车，开放信号",
    },
    {
        "id": "train_0032",
        "asr": "内嵌解锁防盗接车记录",
        "correct": "内勤，解锁3道接车进路",
    },
    {
        "id": "train_0035",
        "asr": "13号道岔现场返回结束。",
        "correct": "2号，13号道岔现场反位解锁",
    },
    {
        "id": "train_0018_v2",
        "asr": "三道空闲，站五掉车，松引，十三号道岔，现场返回，加速。将进入上的非故障道岔，操纵到所需位置。确认进入上的非故障道岔位置，开动正确。",
        "correct": "3道空闲本站无调车作业，13号道岔现场反位加锁。将进路上的非故障道岔操纵到所需位置，确认进路上的非故障道岔位置开通正确",
    },
    {
        "id": "train_0007_v2",
        "asr": "13号到场，我表示现场检查。",
        "correct": "13号道岔无表示现场检查",
    },
    {
        "id": "train_0014_v2",
        "asr": "赤道了，二号四八六幺五次，三道停车，十三号道岔，现场返位加速。",
        "correct": "知道了 2号，48615次3道停车，13号道岔现场反位加锁",
    },
    {
        "id": "train_0009",
        "asr": "登记县城设备检查登记簿，因13号道岔5表份。48615次列车进路，与电务人员沟通，破封开锁开箱。取编号为001、002手摇把2把，及转车机钥匙，并下岗人员代持1号扳道阀。",
        "correct": "报告值班员，登记行车设备检查登记簿，因13号道岔无表示。办理48615次列车进路，与电务人员沟通，破封开锁开箱。取编号为001、002手摇把2把，及转辙机钥匙，并加岗人员带到1号扳道员",
    },
    {
        "id": "train_0041",
        "asr": "值班员13号到厂控制台反馈，恢复表示，电务消极13号到厂，无表示故障已修复。安全节点已合好，经车电双方联合实验，良好。恢复设备正常使用。电务消极，盛行区间逻辑检查，报警故障已修复。经联合实验，良好，恢复设备正常使用。",
        "correct": "报告值班员，13号道岔控制台反位恢复表示。电务销记13号道岔无表示故障已修复，安全节点已合好，经车电双方联合试验良好，恢复设备正常使用。电务销记，上行区间逻辑检查报警故障已修复，经联合实验良好，恢复设备正常使用",
    },
]

# 四种语义精修模式
MODES = ["baseline", "nbest", "rag", "fusion"]

MODE_NAMES = {
    "baseline": "基线LLM",
    "nbest": "N-best融合",
    "rag": "RAG增强",
    "fusion": "融合模式",
}


@pytest.fixture(scope="module")
def pipeline():
    """初始化pipeline."""
    p = PostCorrectionPipeline()
    p.warmup()
    return p


def test_three_directions(pipeline):
    """对比测试四种LLM纠错模式的效果."""
    results = []

    print(f"\n{'='*120}")
    print(f"三方向LLM纠错对比测试 - 共{len(HARD_CASES)}条hard case")
    print(f"{'='*120}")

    # 汇总统计
    mode_stats = {mode: {"total_cer": 0.0, "improved": 0, "degraded": 0, "unchanged": 0, "total_time": 0.0} for mode in MODES}

    for i, case in enumerate(HARD_CASES):
        asr_text = case["asr"]
        correct_text = case["correct"]

        # 规则纠错（Layer 1-3）
        t0 = time.time()
        rule_result = pipeline.run(asr_text, layers=[1, 2, 3])
        rule_time = time.time() - t0

        cer_original = cer(asr_text, correct_text)
        cer_rule = cer(rule_result.corrected, correct_text)

        case_result = {
            "id": case["id"],
            "asr": asr_text,
            "correct": correct_text,
            "rule_corrected": rule_result.corrected,
            "cer_original": round(cer_original, 4),
            "cer_rule": round(cer_rule, 4),
            "rule_time_ms": round(rule_time * 1000, 1),
            "modes": {},
        }

        print(f"\n[{i+1}/{len(HARD_CASES)}] {case['id']}")
        print(f"  ASR原文: {asr_text[:80]}...")
        print(f"  正确文本: {correct_text[:80]}...")
        print(f"  规则纠错: {rule_result.corrected[:80]}...")
        print(f"  CER: 原始={cer_original:.3f} 规则={cer_rule:.3f}")

        # 测试每种LLM模式
        for mode in MODES:
            t0 = time.time()
            try:
                llm_result = pipeline.run(
                    asr_text,
                    layers=[1, 2, 3],
                    enable_semantic=True,
                    semantic_mode=mode,
                )
                llm_time = time.time() - t0

                cer_llm = cer(llm_result.corrected, correct_text)
                _, term_hits, term_total = term_accuracy(llm_result.corrected, correct_text)
                _, entity_hits, entity_total = entity_fidelity(llm_result.corrected, correct_text)

                status = "[改善]" if cer_llm < cer_rule - 0.001 else ("[劣化]" if cer_llm > cer_rule + 0.001 else "[不变]")

                if cer_llm < cer_rule - 0.001:
                    mode_stats[mode]["improved"] += 1
                elif cer_llm > cer_rule + 0.001:
                    mode_stats[mode]["degraded"] += 1
                else:
                    mode_stats[mode]["unchanged"] += 1

                mode_stats[mode]["total_cer"] += cer_llm
                mode_stats[mode]["total_time"] += llm_time

                case_result["modes"][mode] = {
                    "corrected": llm_result.corrected,
                    "cer": round(cer_llm, 4),
                    "status": status,
                    "term_hits": term_hits,
                    "term_total": term_total,
                    "entity_hits": entity_hits,
                    "entity_total": entity_total,
                    "time_ms": round(llm_time * 1000, 1),
                }

                print(f"  {MODE_NAMES[mode]:8s}: CER={cer_llm:.3f} {status} ({llm_time*1000:.0f}ms) {llm_result.corrected[:60]}...")

            except Exception as e:
                llm_time = time.time() - t0
                case_result["modes"][mode] = {
                    "error": f"{type(e).__name__}: {e}",
                    "time_ms": round(llm_time * 1000, 1),
                }
                print(f"  {MODE_NAMES[mode]:8s}: [错误] {type(e).__name__}: {e}")

        results.append(case_result)

    # 汇总统计
    n = len(HARD_CASES)
    print(f"\n{'='*120}")
    print(f"汇总统计（{n}条hard case）")
    print(f"{'='*120}")
    print(f"{'模式':<12s} {'平均CER':>8s} {'改善':>6s} {'劣化':>6s} {'不变':>6s} {'平均耗时':>10s}")
    print(f"{'-'*52}")

    # 计算规则平均CER
    rule_total_cer = sum(r["cer_rule"] for r in results)
    print(f"{'规则纠错':<12s} {rule_total_cer/n:>8.4f} {'':>6s} {'':>6s} {'':>6s} {'':>10s}")

    for mode in MODES:
        stats = mode_stats[mode]
        avg_cer = stats["total_cer"] / n if n else 0
        avg_time = stats["total_time"] / n * 1000 if n else 0
        print(
            f"{MODE_NAMES[mode]:<12s} {avg_cer:>8.4f} "
            f"{stats['improved']:>6d} {stats['degraded']:>6d} {stats['unchanged']:>6d} "
            f"{avg_time:>8.0f}ms"
        )

    # 保存报告
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"three_directions_test_{int(time.time())}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": n,
                "rule_avg_cer": round(rule_total_cer / n, 4),
                "mode_stats": {
                    mode: {
                        "avg_cer": round(stats["total_cer"] / n, 4),
                        "improved": stats["improved"],
                        "degraded": stats["degraded"],
                        "unchanged": stats["unchanged"],
                        "avg_time_ms": round(stats["total_time"] / n * 1000, 1),
                    }
                    for mode, stats in mode_stats.items()
                },
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
