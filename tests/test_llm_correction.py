"""LLM语义层测试：使用改错后仍然错误的case测试LLM纠错效果."""

import json
import time
from pathlib import Path

import pytest

from src.pipeline import PostCorrectionPipeline


# 10条改错后仍然错误的case（从e2e_long_hotwords_report中提取，去重）
LLM_TEST_CASES = [
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
        "id": "train_0003_v2",
        "asr": "丝瓜冬药六次。",
        "correct": "48016次",
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
        "correct": "知道了  2号，48615次3道停车，13号道岔现场反位加锁",
    },
    {
        "id": "train_0009",
        "asr": "登记县城设备检查登记簿，因13号道岔5表份。48615次列车进路，与电务人员沟通，破封开锁开箱。 取编号为001、002手摇把2把，及转车机钥匙，并下岗人员代持1号扳道阀。",
        "correct": "报告值班员，登记行车设备检查登记簿，因13号道岔无表示。办理48615次列车进路，与电务人员沟通，破封开锁开箱。取编号为001、002手摇把2把，及转辙机钥匙，并加岗人员带到1号扳道员",
    },
]


@pytest.fixture(scope="module")
def pipeline():
    """初始化pipeline（启用LLM语义层）."""
    p = PostCorrectionPipeline()
    p.warmup()
    return p


def test_llm_correction(pipeline):
    """测试LLM语义层的纠错效果."""
    from tests.utils.metrics import cer, term_accuracy, entity_fidelity

    results = []
    total_cer_before = 0.0
    total_cer_after_rule = 0.0
    total_cer_after_llm = 0.0

    print(f"\n{'='*100}")
    print(f"LLM语义层测试 - 共{len(LLM_TEST_CASES)}条改错后仍错误的case")
    print(f"{'='*100}")

    for i, case in enumerate(LLM_TEST_CASES):
        asr_text = case["asr"]
        correct_text = case["correct"]

        # 1. 规则纠错（Layer 1-3）
        t0 = time.time()
        rule_result = pipeline.run(asr_text, layers=[1, 2, 3])
        rule_time = time.time() - t0

        # 2. LLM语义纠错（Layer 4）
        t0 = time.time()
        llm_result = pipeline.run(asr_text, layers=[1, 2, 3], enable_semantic=True)
        llm_time = time.time() - t0

        # 计算CER
        cer_original = cer(asr_text, correct_text)
        cer_rule = cer(rule_result.corrected, correct_text)
        cer_llm = cer(llm_result.corrected, correct_text)

        # 计算术语准确率
        _, term_hits, term_total = term_accuracy(llm_result.corrected, correct_text)

        total_cer_before += cer_original
        total_cer_after_rule += cer_rule
        total_cer_after_llm += cer_llm

        status = "[改善]" if cer_llm < cer_rule else ("[劣化]" if cer_llm > cer_rule else "[不变]")
        print(f"\n[{i+1}] {case['id']} {status}")
        print(f"  ASR原文: {asr_text}")
        print(f"  正确文本: {correct_text}")
        print(f"  规则纠错: {rule_result.corrected}")
        print(f"  LLM纠错: {llm_result.corrected}")
        print(f"  CER: 原始={cer_original:.3f} 规则={cer_rule:.3f} LLM={cer_llm:.3f}")
        print(f"  耗时: 规则={rule_time*1000:.0f}ms LLM={llm_time*1000:.0f}ms")

        results.append({
            "id": case["id"],
            "asr": asr_text,
            "correct": correct_text,
            "rule_corrected": rule_result.corrected,
            "llm_corrected": llm_result.corrected,
            "cer_original": round(cer_original, 4),
            "cer_rule": round(cer_rule, 4),
            "cer_llm": round(cer_llm, 4),
            "term_hits": term_hits,
            "term_total": term_total,
            "rule_time_ms": round(rule_time * 1000, 1),
            "llm_time_ms": round(llm_time * 1000, 1),
        })

    n = len(LLM_TEST_CASES)
    avg_cer_before = total_cer_before / n
    avg_cer_after_rule = total_cer_after_rule / n
    avg_cer_after_llm = total_cer_after_llm / n

    print(f"\n{'='*100}")
    print(f"LLM语义层测试汇总")
    print(f"{'='*100}")
    print(f"样本数: {n}")
    print(f"平均CER: 原始={avg_cer_before:.4f} 规则={avg_cer_after_rule:.4f} LLM={avg_cer_after_llm:.4f}")
    print(f"CER改善: 规则={avg_cer_before - avg_cer_after_rule:.4f} LLM={avg_cer_before - avg_cer_after_llm:.4f}")
    print(f"LLM额外改善: {avg_cer_after_rule - avg_cer_after_llm:.4f}")

    improved = sum(1 for r in results if r["cer_llm"] < r["cer_rule"])
    degraded = sum(1 for r in results if r["cer_llm"] > r["cer_rule"])
    unchanged = n - improved - degraded
    print(f"LLM vs 规则: 改善={improved} 不变={unchanged} 劣化={degraded}")

    # 保存报告
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"llm_test_report_{int(time.time())}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": n,
                "avg_cer_before": round(avg_cer_before, 4),
                "avg_cer_after_rule": round(avg_cer_after_rule, 4),
                "avg_cer_after_llm": round(avg_cer_after_llm, 4),
                "rule_improvement": round(avg_cer_before - avg_cer_after_rule, 4),
                "llm_improvement": round(avg_cer_before - avg_cer_after_llm, 4),
                "llm_extra_improvement": round(avg_cer_after_rule - avg_cer_after_llm, 4),
                "improved_count": improved,
                "unchanged_count": unchanged,
                "degraded_count": degraded,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
