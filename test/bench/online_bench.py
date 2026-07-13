"""Online serving benchmark with workload presets.

Usage (Modal):
    modal run test/bench/online_bench.py::load_test --preset balanced
    modal run test/bench/online_bench.py::load_test --preset decode-heavy
    modal run test/bench/online_bench.py::load_test --preset prefill-heavy
    modal run test/bench/online_bench.py::load_test --preset balanced --num-trials 3

Results saved as bench_TIMESTAMP.json; view with:
    python scripts/bench_dashboard.py
"""

import json
import math
import random
import time
from typing import List

import modal

# Workload presets tuned to sit just below saturation (from sweep_20260712_031259).
PRESETS = {
    "decode-heavy": {
        "description": "short input, long output — memory/bandwidth bound",
        "input_min": 64,
        "input_max": 256,
        "output_min": 512,
        "output_max": 2048,
        "arrival_rate": 4.0,
        "decay_rate": 0.0,
        "duration_s": 60.0,
    },
    "balanced": {
        "description": "medium input + output — general workload",
        "input_min": 256,
        "input_max": 1024,
        "output_min": 256,
        "output_max": 1024,
        "arrival_rate": 6.0,
        "decay_rate": 0.0,
        "duration_s": 60.0,
    },
    "prefill-heavy": {
        "description": "long input, short output — compute bound",
        "input_min": 1024,
        "input_max": 2048,
        "output_min": 32,
        "output_max": 128,
        "arrival_rate": 12.0,
        "decay_rate": 0.0,
        "duration_s": 60.0,
    },
}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .pip_install("ninja", "packaging", "wheel", "torch")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("transformers>=4.51.0,<5.0.0", "xxhash", "rich")
    .apt_install(["git", "build-essential"])
    .add_local_dir("nanovllm", "/root/nanovllm")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

app = modal.App("nano-vllm", image=image)
MODEL_STORE_PATH = "/vol/models"
LOGS_PATH = "/vol/logs"
model_volume = modal.Volume.from_name("nano-vllm-models", create_if_missing=False)
logs_volume = modal.Volume.from_name("nano-vllm-logs", create_if_missing=True)


@app.function(gpu="A10", volumes={MODEL_STORE_PATH: model_volume, LOGS_PATH: logs_volume}, timeout=3600)
async def load_test(preset: str = "balanced", num_trials: int = 5, debug: bool = True):
    import os
    from datetime import datetime
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from nanovllm import LLM
    from nanovllm.utils.debug_log import debug_log

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")

    cfg = PRESETS[preset]

    if debug:
        os.environ["NANOVLLM_DEBUG"] = "1"
        os.environ["NANOVLLM_DEBUG_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamp = os.environ.get("NANOVLLM_DEBUG_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

    path = "/vol/models/Qwen_Qwen3-0.6B"
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    print(f"Preset: {preset} — {cfg['description']}")
    print(
        f"  input={cfg['input_min']}–{cfg['input_max']} tok  "
        f"output={cfg['output_min']}–{cfg['output_max']} tok  "
        f"rate={cfg['arrival_rate']} req/s  duration={cfg['duration_s']}s"
    )

    warmup_metrics = None
    trial_results = []

    for i in range(num_trials + 1):  # trial 0 is warmup
        is_warmup = i == 0
        label = "Warmup" if is_warmup else f"Trial {i}/{num_trials}"
        print(f"\n--- {label} ---")

        if debug:
            debug_log({"event": "trial_start", "trial": i, "is_warmup": is_warmup})

        metrics = await run_test(llm, cfg)

        if debug:
            debug_log({"event": "trial_end", "trial": i, "is_warmup": is_warmup})

        print(
            f"{label}: {metrics['throughput']:.1f} tok/s | "
            f"TTFT p50={metrics['p50_ttft']*1000:.1f}ms | "
            f"TPOT={metrics['avg_tpot']*1000:.1f}ms | "
            f"E2E={metrics['avg_e2e']*1000:.1f}ms"
        )

        if is_warmup:
            warmup_metrics = metrics
            print("  (warmup excluded from averages)")
        else:
            trial_results.append(metrics)

    await llm.shutdown()

    console = Console()
    table = Table(title=f"Benchmark Results — {preset}", box=box.ROUNDED, show_lines=True)
    table.add_column("Trial", style="bold")
    table.add_column("Throughput\n(tok/s)", justify="right")
    table.add_column("TTFT p50\n(ms)", justify="right")
    table.add_column("TTFT p99\n(ms)", justify="right")
    table.add_column("TPOT avg\n(ms)", justify="right")
    table.add_column("E2E avg\n(ms)", justify="right")

    def fmt_row(m):
        return (
            f"{m['throughput']:.1f}",
            f"{m['p50_ttft']*1000:.1f}",
            f"{m['p99_ttft']*1000:.1f}",
            f"{m['avg_tpot']*1000:.1f}",
            f"{m['avg_e2e']*1000:.1f}",
        )

    table.add_row("Warmup", *fmt_row(warmup_metrics), style="dim")
    for idx, m in enumerate(trial_results, start=1):
        table.add_row(str(idx), *fmt_row(m))

    avg = {
        "throughput": sum(m["throughput"] for m in trial_results) / len(trial_results),
        "p50_ttft": sum(m["p50_ttft"] for m in trial_results) / len(trial_results),
        "p99_ttft": sum(m["p99_ttft"] for m in trial_results) / len(trial_results),
        "avg_tpot": sum(m["avg_tpot"] for m in trial_results) / len(trial_results),
        "avg_e2e": sum(m["avg_e2e"] for m in trial_results) / len(trial_results),
    }
    table.add_row("AVG", *fmt_row(avg), style="bold")
    console.print(table)

    params = {
        "preset": preset,
        "description": cfg["description"],
        "model": path,
        "gpu": "A10",
        "max_model_len": 4096,
        "arrival_rate": cfg["arrival_rate"],
        "decay_rate": cfg["decay_rate"],
        "duration_s": cfg["duration_s"],
        "input_min": cfg["input_min"],
        "input_max": cfg["input_max"],
        "output_min": cfg["output_min"],
        "output_max": cfg["output_max"],
        "num_trials": num_trials,
        "enforce_eager": False,
    }
    results_path = f"{LOGS_PATH}/bench_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {"timestamp": timestamp, "params": params, "warmup": warmup_metrics, "trials": trial_results},
            f, indent=2,
        )

    logs_volume.commit()
    print(f"\nResults saved to bench_{timestamp}.json")

    return avg["throughput"]


async def run_test(llm, cfg: dict):
    import asyncio
    from random import randint, seed
    from nanovllm import SamplingParams
    from nanovllm.utils.debug_log import debug_log

    seed(0)

    interarrival_times = generate_interarrival_times(
        arrival_rate=cfg["arrival_rate"],
        decay_rate=cfg["decay_rate"],
        duration=cfg["duration_s"],
    )

    tasks = []
    start_time = time.perf_counter()

    for i, gap in enumerate(interarrival_times):
        await asyncio.sleep(gap)
        input_len = randint(cfg["input_min"], cfg["input_max"])
        output_len = randint(cfg["output_min"], cfg["output_max"])
        prompt_token_ids = [randint(0, 10000) for _ in range(input_len)]
        sampling_params = SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=output_len,
        )
        debug_log({"event": "request_arrive", "req_idx": i})
        tasks.append(asyncio.create_task(llm.agenerate([prompt_token_ids], sampling_params)))

    all_results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_time
    flat = [r[0] for r in all_results]

    total_output_tokens = sum(r["num_output_tokens"] for r in flat)
    throughput = total_output_tokens / elapsed

    ttfts = [r["ttft"] for r in flat if r["ttft"] is not None]
    e2es = [r["e2e_latency"] for r in flat if r["e2e_latency"] is not None]
    tpots = [
        (r["e2e_latency"] - r["ttft"]) / (r["num_output_tokens"] - 1)
        for r in flat
        if r["ttft"] is not None and r["e2e_latency"] is not None and r["num_output_tokens"] > 1
    ]

    def percentile(data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = max(0, min(int(len(sorted_data) * p / 100 + 0.5) - 1, len(sorted_data) - 1))
        return sorted_data[idx]

    metrics = {
        "throughput": throughput,
        "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "p50_ttft": percentile(ttfts, 50),
        "p99_ttft": percentile(ttfts, 99),
        "avg_tpot": sum(tpots) / len(tpots) if tpots else 0.0,
        "avg_e2e": sum(e2es) / len(e2es) if e2es else 0.0,
        "num_requests": len(flat),
        "total_output_tokens": total_output_tokens,
        "elapsed": elapsed,
    }

    print(
        f"  {len(flat)} reqs | {total_output_tokens} tok | {elapsed:.2f}s | "
        f"{throughput:.1f} tok/s"
    )

    return metrics


def generate_interarrival_times(arrival_rate: float, decay_rate: float, duration: float) -> List[float]:
    interarrival_times = []
    t = 0.0
    while t < duration:
        current_rate = arrival_rate * math.exp(-decay_rate * t)
        gap = random.expovariate(current_rate)
        if t + gap >= duration:
            break
        t += gap
        interarrival_times.append(gap)
    return interarrival_times


if __name__ == "__main__":
    with app.run():
        load_test.remote(preset="balanced", num_trials=5)
