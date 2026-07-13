# Performance Changelog

Tracks engine changes and their effect on serving performance. Each stage is measured with the same sweep harness (`test/bench/sweep_bench.py`).

**Test config:** Modal A10 · Qwen3-0.6B · `max_model_len=4096` · CUDA graphs · Poisson arrivals · rates 1–48 req/s · [sweep_20260712_031259](logs/sweep_20260712_031259.json)

| Stage | Summary | Prefill knee | Balanced knee | Decode knee |
|-------|---------|-------------|---------------|-------------|
| 0 | Async serving + threaded `step()` | 24 req/s · 923 tok/s | 8 req/s · 2090 tok/s | 6 req/s · fails |

*knee = first saturated arrival rate · tok/s = peak throughput at rate just below knee*

---

## Stage 0 — Async serving baseline

**Commit:** `44958fa` · **Date:** 2026-07-11

### Changes

| Component | What changed |
|-----------|--------------|
| `llm_engine.py` | `agenerate()` + background `_engine_loop` continuously runs the scheduler |
| `llm_engine.py` | `asyncio.to_thread(self.step)` — GPU forward pass off the event loop so new requests arrive during inference |
| `llm_engine.py` | Request grouping via UUID + futures (multi-prompt batches) |
| `sequence.py` | `submit_time` / `first_token_time` / `finish_time` for TTFT & E2E (instrumentation) |

Baseline inherits upstream scheduler (prefill-priority, FCFS decode, preemption on KV pressure), paged KV cache, prefix caching, and CUDA graphs unchanged.

### Sweep results

**Prefill-heavy** (1024–2048 in · 32–128 out)

| req/s | tok/s | p50 TTFT | p50 E2E | |
|------:|------:|---------:|--------:|---|
| 12 | 815 | 51 ms | 2.3 s | ← online preset (12 req/s) |
| 16 | 923 | 2.0 s | 8.6 s | peak throughput |
| 24 | 879 | 17.3 s | 24.3 s | **knee** — throughput plateau |

**Balanced** (256–1024 in · 256–1024 out)

| req/s | tok/s | p50 TTFT | p50 E2E | |
|------:|------:|---------:|--------:|---|
| 6 | 2090 | 48 ms | 24.1 s | ← online preset (6 req/s) |
| 8 | 2057 | 86 ms | 40.2 s | **knee** — throughput plateau, p99 TTFT 20.7 s |

**Decode-heavy** (64–256 in · 512–2048 out)

| req/s | tok/s | p50 TTFT | p50 E2E | |
|------:|------:|---------:|--------:|---|
| 4 | 2070 | 46 ms | 40.6 s | ← online preset (4 req/s) |
| 6 | 0 | — | — | **knee** — drain timeout, 0% completions (172 pending) |

### Takeaways

- Threading `step()` unlocks concurrent request intake but the **scheduler and KV capacity** are the bottlenecks, not the async wrapper.
- Decode-heavy saturates hardest — long outputs exhaust KV blocks; engine fails catastrophically at 6 req/s rather than graceful degradation.
- Prefill-heavy has the highest req/s ceiling (24) but TTFT degrades sharply above 16 req/s.
- Balanced peaks at ~2090 tok/s; latency blowout (40 s E2E) precedes throughput collapse.
