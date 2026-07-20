"""性能基准测试.

测试维度:
1. Latency: 单条处理耗时及分位值
2. Memory: 加载内存和峰值内存
3. Throughput: 批量处理QPS

Usage:
    python tests/benchmark.py
    python tests/benchmark.py --samples 200 --output-dir reports/
"""

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PostCorrectionPipeline


TEST_TEXTS = [
    # 短文本 (~20字)
    "18号道岔开通反位",
    # 中等文本 (~50字)
    "2号扳道员，48015次1道发车，将18号道岔开通1道并加锁，确认信号开放",
    # 长文本 (~100字)
    "按照列车运行计划，核对车次、时刻、命令、指示，必要时与列车调度员联系，"
    "确认接车进路正确，道岔位置正确并锁闭，信号机状态正常，"
    "股道空闲无异常，准备接车。",
    # 复杂文本 (~150字，含多种实体)
    "G1023次列车预告，3道接车，18号道岔开通定位并加锁，"
    "出站信号机开放，限速60km/h，K123+456处注意运行，"
    "2号扳道员确认进路正确，咽喉区无异常，闭塞分区空闲，"
    "与邻站办理闭塞手续，同意发车。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="性能基准测试")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="每条文本的重复测试次数 (默认: 100)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 10, 50],
        help="批量测试的批次大小 (默认: 1 10 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="报告输出目录",
    )
    return parser.parse_args()


def measure_latency(pipeline: PostCorrectionPipeline, texts: List[str], n: int) -> dict:
    """测量单条处理延迟."""
    print("[BENCH] 测量单条延迟...")
    latencies = []

    # 预热
    for text in texts:
        pipeline.run(text, layers=[1, 2, 3])

    # 测试
    for _ in range(n):
        for text in texts:
            t0 = time.perf_counter()
            pipeline.run(text, layers=[1, 2, 3])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    return {
        "count": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(p50, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
    }


def measure_memory(pipeline: PostCorrectionPipeline) -> dict:
    """测量内存占用.

    注意：tracemalloc 仅追踪 Python 层内存分配，不计 C 扩展（如 jieba 内部）。
    实际内存占用可能高于报告值。如需精确测量，建议使用 memory_profiler 或 psutil。
    """
    print("[BENCH] 测量内存占用...")

    # 测量加载后的内存基线
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # 运行一些文本触发可能的内部缓存
    for text in TEST_TEXTS:
        pipeline.run(text, layers=[1, 2, 3])

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

    return {
        "peak_kb": round(total_diff / 1024, 2),
        "top_allocations": [
            {
                "file": str(stat.traceback.format()[-1]) if stat.traceback.format() else "unknown",
                "size_kb": round(stat.size_diff / 1024, 2),
            }
            for stat in top_stats[:5]
            if stat.size_diff > 0
        ],
    }


def measure_throughput(
    pipeline: PostCorrectionPipeline,
    texts: List[str],
    batch_sizes: List[int],
) -> dict:
    """测量批量处理吞吐量."""
    print("[BENCH] 测量吞吐量...")
    results = {}

    for batch_size in batch_sizes:
        batch = (texts * ((batch_size // len(texts)) + 1))[:batch_size]

        # 预热
        pipeline.run_batch(batch[:1], layers=[1, 2, 3])

        t0 = time.perf_counter()
        pipeline.run_batch(batch, layers=[1, 2, 3])
        t1 = time.perf_counter()

        elapsed = t1 - t0
        qps = batch_size / elapsed if elapsed > 0 else 0
        avg_latency_ms = (elapsed / batch_size) * 1000

        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "total_time_ms": round(elapsed * 1000, 2),
            "qps": round(qps, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
        }

    return results


def measure_by_length(pipeline: PostCorrectionPipeline) -> dict:
    """按文本长度测量延迟."""
    print("[BENCH] 按文本长度测量...")
    lengths = [20, 50, 100, 150, 200]
    results = {}

    for length in lengths:
        # 构造对应长度的文本
        text = "18号道岔开通反位，信号好了。" * (length // 20)
        text = text[:length]

        # 预热
        pipeline.run(text, layers=[1, 2, 3])

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            pipeline.run(text, layers=[1, 2, 3])
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results[f"len_{length}"] = {
            "length": len(text),
            "mean_ms": round(statistics.mean(times), 2),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
        }

    return results


def main():
    args = parse_args()

    print("=" * 50)
    print("ASR纠错系统性能基准测试")
    print("=" * 50)

    pipeline = PostCorrectionPipeline()
    pipeline.warmup()

    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples_per_text": args.samples,
        },
        "latency": measure_latency(pipeline, TEST_TEXTS, args.samples),
        "memory": measure_memory(pipeline),
        "throughput": measure_throughput(pipeline, TEST_TEXTS, args.batch_sizes),
        "by_length": measure_by_length(pipeline),
    }

    # 保存报告
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"benchmark_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "=" * 50)
    print("性能基准摘要")
    print("=" * 50)
    lat = report["latency"]
    print(f"单条延迟: mean={lat['mean_ms']:.1f}ms, p50={lat['p50_ms']:.1f}ms, "
          f"p95={lat['p95_ms']:.1f}ms, p99={lat['p99_ms']:.1f}ms")
    mem = report["memory"]
    print(f"峰值内存: {mem['peak_kb']:.1f}KB")
    for key, val in report["throughput"].items():
        print(f"吞吐量 ({key}): {val['qps']:.1f} qps, 平均延迟 {val['avg_latency_ms']:.1f}ms")
    print("=" * 50)
    print(f"[INFO] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
