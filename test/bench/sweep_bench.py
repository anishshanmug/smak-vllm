"""Saturation sweep benchmark.

Runs three workload profiles (prefill-heavy, decode-heavy, balanced) each
sweeping arrival rates with uniform Poisson arrivals to map the throughput
ceiling and latency knee of the engine.

Design goals:
  - Bounded drain time so long decode workloads cannot hang the sweep
  - Saturation detection that catches decode-bound overload (E2E, drain timeout)
  - Incremental checkpointing after every rate point

Usage (Modal):
    modal run test/bench/sweep_bench.py
    modal run -d test/bench/sweep_bench.py
    modal run test/bench/sweep_bench.py::sweep_test --rates '[1,2,4,8,16,32]'

Results are saved as sweep_TIMESTAMP.json to the nano-vllm-logs volume and
can be viewed via:
    python scripts/bench_dashboard.py
"""

import json
import random
import time

import modal

PROFILES = [
    {
        "label": "prefill-heavy",
        "description": "long input, short output — compute bound",
        "input_min": 1024,
        "input_max": 2048,
        "output_min": 32,
        "output_max": 128,
    },
    {
        "label": "balanced",
        "description": "medium input + output — general workload",
        "input_min": 256,
        "input_max": 1024,
        "output_min": 256,
        "output_max": 1024,
    },
    {
        "label": "decode-heavy",
        "description": "short input, long output — memory/bandwidth bound",
        "input_min": 64,
        "input_max": 256,
        "output_min": 512,
        "output_max": 2048,
    },
]

DEFAULT_RATES = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]

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

app = modal.App("nano-vllm-sweep", image=image)
MODEL_STORE_PATH = "/vol/models"
LOGS_PATH = "/vol/logs"
model_volume = modal.Volume.from_name("nano-vllm-models", create_if_missing=False)
logs_volume = modal.Volume.from_name("nano-vllm-logs", create_if_missing=True)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100 + 0.5) - 1, len(s) - 1))
    return s[idx]


def _build_metrics(flat: list[dict], num_submitted: int, elapsed: float,
                   drain_timed_out: bool, num_pending: int) -> dict:
    total_output_tokens = sum(r["num_output_tokens"] for r in flat)
    ttfts = [r["ttft"] for r in flat if r["ttft"] is not None]
    e2es = [r["e2e_latency"] for r in flat if r["e2e_latency"] is not None]
    tpots = [
        (r["e2e_latency"] - r["ttft"]) / (r["num_output_tokens"] - 1)
        for r in flat
        if r["ttft"] is not None and r["e2e_latency"] is not None and r["num_output_tokens"] > 1
    ]

    return {
        "throughput": total_output_tokens / elapsed if elapsed > 0 else 0.0,
        "p50_ttft": _percentile(ttfts, 50),
        "p99_ttft": _percentile(ttfts, 99),
        "p50_e2e": _percentile(e2es, 50),
        "p99_e2e": _percentile(e2es, 99),
        "avg_tpot": sum(tpots) / len(tpots) if tpots else 0.0,
        "avg_e2e": sum(e2es) / len(e2es) if e2es else 0.0,
        "num_requests": len(flat),
        "num_submitted": num_submitted,
        "num_pending": num_pending,
        "total_output_tokens": total_output_tokens,
        "elapsed": elapsed,
        "drain_timed_out": drain_timed_out,
    }


async def _run_uniform(
    llm,
    arrival_rate: float,
    arrival_duration: float,
    drain_timeout: float,
    profile: dict,
    *,
    wait_for_drain: bool = True,
):
    """Submit Poisson arrivals for `arrival_duration`, then drain up to `drain_timeout`.

    If drain times out, returns partial metrics from completed requests only.
    """
    import asyncio
    from random import randint, seed
    from nanovllm import SamplingParams

    seed(42)
    tasks: list[asyncio.Task] = []
    num_submitted = 0
    start_time = time.perf_counter()
    t = 0.0

    while t < arrival_duration:
        gap = random.expovariate(arrival_rate)
        if t + gap >= arrival_duration:
            break
        await asyncio.sleep(gap)
        t += gap
        input_len = randint(profile["input_min"], profile["input_max"])
        output_len = randint(profile["output_min"], profile["output_max"])
        prompt_token_ids = [randint(0, 10000) for _ in range(input_len)]
        sampling_params = SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=output_len,
        )
        tasks.append(asyncio.create_task(llm.agenerate([prompt_token_ids], sampling_params)))
        num_submitted += 1

    arrival_end = time.perf_counter()
    drain_timed_out = False
    num_pending = 0
    flat: list[dict] = []

    if wait_for_drain and tasks:
        done, pending = await asyncio.wait(set(tasks), timeout=drain_timeout)
        for task in done:
            try:
                result = await task
                flat.append(result[0])
            except Exception:
                pass
        for task in pending:
            task.cancel()
        num_pending = len(pending)
        drain_timed_out = num_pending > 0
    elif tasks:
        # Warmup: don't wait for completions — just let arrivals run.
        num_pending = len(tasks)

    elapsed = time.perf_counter() - start_time
    metrics = _build_metrics(flat, num_submitted, elapsed, drain_timed_out, num_pending)
    metrics["arrival_elapsed"] = arrival_end - start_time
    metrics["drain_timeout"] = drain_timeout
    return metrics


def _is_saturated(metrics: dict, prev_throughput: float | None) -> tuple[bool, str]:
    """Return (saturated, reason)."""
    completion = metrics["num_requests"] / max(metrics["num_submitted"], 1)

    if completion < 0.8:
        return True, f"completion {completion:.0%} < 80%"

    if metrics["drain_timed_out"]:
        return True, f"drain timed out ({metrics['num_pending']} requests still pending)"

    if metrics["p99_ttft"] > 30.0:
        return True, f"TTFT p99 {metrics['p99_ttft']*1000:.0f}ms > 30s"

    if metrics["p99_e2e"] > 120.0:
        return True, f"E2E p99 {metrics['p99_e2e']*1000:.0f}ms > 120s"

    if prev_throughput and prev_throughput > 0:
        gain = metrics["throughput"] / prev_throughput
        if gain < 1.05:
            return True, f"throughput plateau ({metrics['throughput']:.0f} vs {prev_throughput:.0f} tok/s)"

    return False, ""


def _save_checkpoint(
    timestamp: str,
    params: dict,
    all_profile_results: list,
    *,
    status: str = "in_progress",
    current_profile: str | None = None,
):
    sweep_data = {
        "timestamp": timestamp,
        "type": "sweep",
        "status": status,
        "current_profile": current_profile,
        "params": params,
        "profiles": all_profile_results,
        "results": all_profile_results[0]["results"] if all_profile_results else [],
    }
    results_path = f"{LOGS_PATH}/sweep_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    logs_volume.commit()
    print(f"  [checkpoint] saved {results_path} ({status})")


@app.function(
    gpu="A10",
    volumes={MODEL_STORE_PATH: model_volume, LOGS_PATH: logs_volume},
    timeout=14400,
)
async def sweep_test(
    rates: str = "",
    warmup_duration: float = 15.0,
    measure_duration: float = 30.0,
    warmup_drain_timeout: float = 30.0,
    measure_drain_timeout: float = 120.0,
):
    """Run a multi-profile saturation sweep with bounded drain and checkpointing."""
    from datetime import datetime
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from nanovllm import LLM

    parsed_rates = json.loads(rates) if rates.strip() else DEFAULT_RATES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "/vol/models/Qwen_Qwen3-0.6B"
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    params = {
        "model": path,
        "gpu": "A10",
        "max_model_len": 4096,
        "warmup_duration_s": warmup_duration,
        "measure_duration_s": measure_duration,
        "warmup_drain_timeout_s": warmup_drain_timeout,
        "measure_drain_timeout_s": measure_drain_timeout,
        "decay_rate": 0.0,
        "enforce_eager": False,
        "sweep_rates": parsed_rates,
    }

    console = Console()
    all_profile_results = []
    _save_checkpoint(timestamp, params, all_profile_results, status="started")

    try:
        for profile in PROFILES:
            label = profile["label"]
            console.rule(f"[bold cyan]Profile: {label}[/bold cyan]")
            print(f"  {profile['description']}")
            print(f"  input={profile['input_min']}–{profile['input_max']} tok  "
                  f"output={profile['output_min']}–{profile['output_max']} tok\n")

            rate_results = []
            saturated = False
            prev_throughput = None

            for rate in parsed_rates:
                if saturated:
                    print(f"  [{label}] {rate} req/s: skipped (saturated)")
                    rate_results.append({"arrival_rate": rate, "saturated": True})
                    continue

                print(f"  [{label}] warmup  {rate} req/s ({warmup_duration}s arrivals) ...")
                await _run_uniform(
                    llm, rate, warmup_duration, warmup_drain_timeout, profile,
                    wait_for_drain=False,
                )

                print(f"  [{label}] measure {rate} req/s "
                      f"({measure_duration}s arrivals, {measure_drain_timeout}s drain cap) ...")
                m = await _run_uniform(
                    llm, rate, measure_duration, measure_drain_timeout, profile,
                    wait_for_drain=True,
                )
                m["arrival_rate"] = rate
                m["saturated"] = False
                rate_results.append(m)

                completion = m["num_requests"] / max(m["num_submitted"], 1)
                drain_note = f" | drain_timeout" if m["drain_timed_out"] else ""
                print(
                    f"    -> {m['throughput']:.1f} tok/s | "
                    f"TTFT p99={m['p99_ttft']*1000:.0f}ms | "
                    f"E2E p99={m['p99_e2e']*1000:.0f}ms | "
                    f"done={m['num_requests']}/{m['num_submitted']} ({completion:.0%})"
                    f"{drain_note}"
                )

                is_sat, reason = _is_saturated(m, prev_throughput)
                if is_sat:
                    m["saturated"] = True
                    m["saturation_reason"] = reason
                    print(f"  *** [{label}] saturated at {rate} req/s: {reason} ***\n")
                    saturated = True
                prev_throughput = m["throughput"]

                # Checkpoint after every rate point
                _checkpoint_profiles = list(all_profile_results)
                _checkpoint_profiles.append({"profile": profile, "results": list(rate_results)})
                _save_checkpoint(timestamp, params, _checkpoint_profiles,
                                 status="in_progress", current_profile=label)

            all_profile_results.append({"profile": profile, "results": rate_results})

            table = Table(title=f"{label} — {profile['description']}", box=box.SIMPLE_HEAVY)
            table.add_column("Rate", justify="right")
            table.add_column("Throughput\n(tok/s)", justify="right")
            table.add_column("TTFT p99\n(ms)", justify="right")
            table.add_column("E2E p99\n(ms)", justify="right")
            table.add_column("Done %", justify="right")
            table.add_column("Note", justify="left")
            for r in rate_results:
                if r.get("saturated") and "throughput" not in r:
                    table.add_row(str(r["arrival_rate"]), "—", "—", "—", "skip", "", style="dim")
                else:
                    pct = f"{r['num_requests']/max(r['num_submitted'],1)*100:.0f}%"
                    note = r.get("saturation_reason", "")
                    if r.get("drain_timed_out") and not note:
                        note = "drain timeout"
                    style = "bold red" if r.get("saturated") else ""
                    table.add_row(
                        str(r["arrival_rate"]),
                        f"{r['throughput']:.1f}",
                        f"{r['p99_ttft']*1000:.0f}",
                        f"{r['p99_e2e']*1000:.0f}",
                        pct,
                        note[:30],
                        style=style,
                    )
            console.print(table)

        await llm.shutdown()
        _save_checkpoint(timestamp, params, all_profile_results, status="complete")
        print(f"\nSweep complete — sweep_{timestamp}.json")

    except Exception as e:
        _save_checkpoint(timestamp, params, all_profile_results, status=f"failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    detach = "--detach" in sys.argv or "-d" in sys.argv

    with app.run(detach=detach):
        if detach:
            sweep_test.spawn()
            print("Sweep submitted in detached mode.")
            print("Results checkpoint to sweep_*.json after every rate point.")
            print("Track at: https://modal.com/apps/")
        else:
            sweep_test.remote()
