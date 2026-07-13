# nano-vllm (fork)

A readable vLLM-style inference engine, extended to **evaluate how scheduling, memory, and serving decisions affect performance** under realistic workloads.

This is a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a ~1,700-line educational reimplementation of vLLM's core ideas (paged KV cache, continuous batching, prefix caching, CUDA graphs). The upstream codebase is small enough to read and modify. This fork adds async serving, per-request latency tracking, Poisson-arrival load tests, saturation sweeps, and a dashboard so you can change a mechanism and measure the impact on throughput and latency.

> Educational codebase, not production-ready. Benchmarks run on Modal A10 with Qwen3-0.6B.

## What this fork adds

- **Async online serving** — `agenerate()` with a background scheduler loop; GPU steps run in a thread pool so requests can arrive during inference
- **Latency instrumentation** — per-request TTFT, TPOT, and E2E tracked in the engine
- **Realistic load testing** — Poisson arrivals with decode-heavy, balanced, and prefill-heavy presets
- **Saturation sweeps** — find throughput knees before running steady-state benchmarks
- **Observability** — structured debug logs (scheduler state, KV blocks, preemptions, CUDA memory)
- **Modal workflow** — GPU benchmarks on A10 with persistent model and log volumes
- **Dashboard** — local Chart.js web UI for bench results and scheduler time-series

## Quick start

### Offline (batch)

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(
    ["Hello"],
    SamplingParams(temperature=0.6, max_tokens=256),
)
outputs[0]["text"]
```

### Online (async)

```python
results = await llm.agenerate(
    ["Hello"],
    SamplingParams(temperature=0.6, max_tokens=256),
)
results[0]["ttft"]         # time to first token (seconds)
results[0]["e2e_latency"]  # end-to-end latency (seconds)
results[0]["text"]
```

## Running on Modal

Most benchmarking happens on Modal — no local GPU required.

```bash
poetry install

# Smoke test + download model to volume (first run only)
modal run test/bench/offline_example.py

# Offline throughput
modal run test/bench/offline_bench.py

# Online load test (steady-state, sub-saturation)
modal run test/bench/online_bench.py::load_test --preset balanced

# Saturation sweep (find throughput knee)
modal run test/bench/sweep_bench.py

# Visualize results locally
python scripts/bench_dashboard.py
python scripts/bench_dashboard.py --sweep
```

Modal volumes:
- `nano-vllm-models` — model weights (Qwen3-0.6B)
- `nano-vllm-logs` — bench results (`bench_*.json`), sweep results (`sweep_*.json`), debug logs

## Benchmarking

The evaluation workflow is: **sweep → pick a sub-saturation rate → online bench → dashboard**.

### 1. Find saturation limits

Sweep arrival rates across three workload profiles to map the throughput ceiling and latency knee:

```bash
modal run test/bench/sweep_bench.py
```

Results are checkpointed after every rate point and saved as `sweep_TIMESTAMP.json`.

### 2. Run steady-state load tests

Use a preset tuned to sit just below saturation (rates derived from sweep data):

```bash
modal run test/bench/online_bench.py::load_test --preset decode-heavy
modal run test/bench/online_bench.py::load_test --preset balanced
modal run test/bench/online_bench.py::load_test --preset prefill-heavy
```

| Preset | What it stresses | Arrival rate |
|--------|------------------|--------------|
| decode-heavy | KV memory / bandwidth — short input, long output | ~4 req/s |
| balanced | General mix — medium input and output | ~6 req/s |
| prefill-heavy | Compute / attention — long input, short output | ~12 req/s |

Each run records per-trial TTFT, TPOT, E2E, and throughput. Warmup trial is excluded from averages.

### 3. Visualize

```bash
python scripts/bench_dashboard.py                              # latest bench run
python scripts/bench_dashboard.py --timestamp 20260712_031259  # specific run
python scripts/bench_dashboard.py --sweep                      # latest sweep
```

### What you can evaluate

| Mechanism | Where it lives | Metrics |
|-----------|----------------|---------|
| Continuous batching / scheduling | `nanovllm/engine/scheduler.py` | Throughput, queue depth, preemption rate |
| Paged KV cache / block allocation | `nanovllm/engine/block_manager.py` | KV block utilization, OOM events |
| Prefill vs decode batching | `scheduler.py`, `model_runner.py` | TTFT, TPOT, prefill/decode throughput |
| Async serving / request concurrency | `nanovllm/engine/llm_engine.py` | E2E latency under Poisson arrivals |
| CUDA graphs, prefix caching, TP | inherited from upstream | Offline throughput |

Change a mechanism, re-run the sweep or online bench, and compare results in the dashboard. See [`PERFORMANCE.md`](PERFORMANCE.md) for stage-by-stage changes and sweep results.

## Observability

Enable structured debug logging during a bench run:

```bash
modal run test/bench/online_bench.py::load_test --preset balanced --debug
```

Or set `NANOVLLM_DEBUG=1` for local runs.

Logs capture scheduler events (queue depths, KV block usage, preemptions) and model-runner CUDA memory at each forward pass. Pull logs from Modal:

```bash
modal volume get nano-vllm-logs stage0_debug_*.log logs/
```

View scheduler time-series alongside bench metrics in the dashboard.

## Architecture & learning

- [`mermaid_diagrams.md`](mermaid_diagrams.md) — visual architecture: execution flow, memory management, attention computation
- [`nano_vllm_execution_flow.md`](nano_vllm_execution_flow.md) — detailed code-level execution trace

## Installation

This fork is not published to PyPI. Use Poetry for local development:

```bash
poetry install
```

FlashAttention is installed via prebuilt wheel in the Modal image, not as a Poetry dependency. For local GPU use, install it separately.

### Model weights

Download manually:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Or use Modal — `test/bench/offline_example.py` downloads Qwen3-0.6B into the `nano-vllm-models` volume on first run.

## Inherited from upstream

Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)'s core engine:

- Paged KV cache and prefix caching
- Continuous batching with preemption
- Tensor parallelism
- Torch compilation and CUDA graphs

See the [upstream README](https://github.com/GeeeekExplorer/nano-vllm) for the original design goals and offline throughput comparisons against vLLM.

## Acknowledgements

Based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by GeeeekExplorer.
