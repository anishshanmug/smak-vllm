"""
Benchmark dashboard — fetches results from Modal volume and serves a local webapp.

Usage:
    python scripts/bench_dashboard.py                              # latest bench run
    python scripts/bench_dashboard.py --timestamp 20260710_171500    # specific bench run
    python scripts/bench_dashboard.py --sweep                        # latest sweep_*.json
    python scripts/bench_dashboard.py --sweep --timestamp 20260712_010459
    python scripts/bench_dashboard.py --port 8787
"""

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import modal


LOGS_VOLUME = "nano-vllm-logs"
PORT = 8787


# ---------------------------------------------------------------------------
# Modal volume helpers
# ---------------------------------------------------------------------------

def _list_sweep_files(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if re.match(r"sweep_\d{8}_\d{6}\.json", p))


def _pick_sweep_file(paths: list[str], timestamp: str | None) -> str:
    sweep_files = _list_sweep_files(paths)
    if not sweep_files:
        raise FileNotFoundError("No sweep_*.json files found in nano-vllm-logs volume.")

    if timestamp:
        sweep_path = f"sweep_{timestamp}.json"
        if sweep_path not in sweep_files:
            raise FileNotFoundError(
                f"{sweep_path} not found. Available: {sweep_files}"
            )
        return sweep_path
    return sweep_files[-1]


def fetch_sweep_only(timestamp: str | None) -> dict:
    """Load only a sweep_TIMESTAMP.json from the Modal logs volume."""
    volume = modal.Volume.from_name(LOGS_VOLUME)
    paths = [e.path for e in volume.listdir("/")]
    sweep_path = _pick_sweep_file(paths, timestamp)
    print(f"Fetching {sweep_path} ...")
    sweep_bytes = b"".join(volume.read_file(sweep_path))
    return json.loads(sweep_bytes)


def fetch_volume_files(timestamp: str | None) -> tuple[dict, list[dict], dict | None]:
    """Pull bench JSON + debug log (+ optional sweep JSON) from the Modal logs volume.

    Returns (bench_data, debug_events, sweep_data).
    sweep_data is loaded from sweep_*.json (matching timestamp if given, else latest).
    """
    volume = modal.Volume.from_name(LOGS_VOLUME)
    entries = volume.listdir("/")
    paths = [e.path for e in entries]

    bench_path, debug_path = _pick_files(paths, timestamp)

    print(f"Fetching {bench_path} ...")
    bench_bytes = b"".join(volume.read_file(bench_path))
    bench_data = json.loads(bench_bytes)

    print(f"Fetching {debug_path} ...")
    debug_bytes = b"".join(volume.read_file(debug_path))
    debug_events = _parse_debug_log(debug_bytes.decode())

    sweep_data = None
    sweep_files = _list_sweep_files(paths)
    if sweep_files:
        # Prefer sweep file matching --timestamp, else bench run timestamp, else latest
        sweep_ts = timestamp or bench_data.get("timestamp")
        sweep_path = f"sweep_{sweep_ts}.json" if sweep_ts else None
        if sweep_path and sweep_path in sweep_files:
            pass
        else:
            sweep_path = sweep_files[-1]
        print(f"Fetching {sweep_path} ...")
        sweep_bytes = b"".join(volume.read_file(sweep_path))
        sweep_data = json.loads(sweep_bytes)

    return bench_data, debug_events, sweep_data


def _pick_files(paths: list[str], timestamp: str | None) -> tuple[str, str]:
    bench_files = sorted(p for p in paths if re.match(r"bench_\d{8}_\d{6}\.json", p))
    debug_files = sorted(p for p in paths if re.match(r"stage0_debug_\d{8}_\d{6}\.log", p))

    if not bench_files:
        raise FileNotFoundError("No bench_*.json files found in nano-vllm-logs volume.")
    if not debug_files:
        raise FileNotFoundError("No stage0_debug_*.log files found in nano-vllm-logs volume.")

    if timestamp:
        bench_path = f"bench_{timestamp}.json"
        debug_path = f"stage0_debug_{timestamp}.log"
        if bench_path not in bench_files:
            raise FileNotFoundError(f"{bench_path} not found. Available: {bench_files}")
        if debug_path not in debug_files:
            raise FileNotFoundError(f"{debug_path} not found. Available: {debug_files}")
    else:
        bench_path = bench_files[-1]
        # Match debug log to same timestamp as bench file
        ts = re.search(r"(\d{8}_\d{6})", bench_path).group(1)
        debug_path = f"stage0_debug_{ts}.log"
        if debug_path not in debug_files:
            debug_path = debug_files[-1]

    return bench_path, debug_path


def _parse_debug_log(text: str) -> list[dict]:
    events = []
    prefix = "[nanovllm-debug] "
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            obj = json.loads(line[len(prefix):])
            evt = obj.get("event", "")
            if evt in ("schedule", "preempt", "trial_start", "trial_end",
                       "pre_forward", "post_forward", "request_arrive"):
                events.append(obj)
        except json.JSONDecodeError:
            pass
    return events


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nano-vllm Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }
  h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 28px; }
  h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; color: #cbd5e1; }

  /* --- Metrics table --- */
  .table-wrap { overflow-x: auto; margin-bottom: 36px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { background: #1e293b; color: #94a3b8; text-align: right; padding: 10px 14px;
       font-weight: 600; border-bottom: 1px solid #334155; white-space: nowrap; }
  th:first-child { text-align: left; }
  td { padding: 10px 14px; border-bottom: 1px solid #1e293b; text-align: right; }
  td:first-child { text-align: left; font-weight: 500; }
  tr.warmup td { color: #64748b; }
  tr.avg td { background: #1e293b; font-weight: 700; color: #e2e8f0; }
  tr:not(.warmup):not(.avg):hover td { background: #1e293b55; }

  /* --- Charts grid --- */
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
                 gap: 24px; margin-bottom: 36px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; }
  .card h3 { font-size: 0.9rem; font-weight: 600; color: #94a3b8; margin-bottom: 14px; }
  canvas { width: 100% !important; }

  /* --- Params bar --- */
  .params-bar { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 28px;
                background: #1e293b; border-radius: 8px; padding: 12px 16px; }
  .param { font-size: 0.8rem; color: #94a3b8; }
  .param span { color: #e2e8f0; font-weight: 500; margin-left: 4px; }

  /* --- Trial selector --- */
  .trial-selector { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .trial-btn { background: #334155; border: none; color: #cbd5e1; padding: 6px 14px;
               border-radius: 6px; cursor: pointer; font-size: 0.82rem; transition: background 0.15s; }
  .trial-btn:hover, .trial-btn.active { background: #3b82f6; color: #fff; }
</style>
</head>
<body>

<h1>nano-vllm Benchmark Dashboard</h1>
<div class="subtitle">Run: <code id="run-ts"></code></div>

<div class="params-bar" id="params-bar"></div>

<div id="bench-section">
<h2>Metrics Summary</h2>
<div class="table-wrap">
<table id="metrics-table">
  <thead>
    <tr>
      <th>Trial</th>
      <th>Throughput<br>(tok/s)</th>
      <th>TTFT p50<br>(ms)</th>
      <th>TTFT p99<br>(ms)</th>
      <th>TPOT avg<br>(ms)</th>
      <th>E2E avg<br>(ms)</th>
    </tr>
  </thead>
  <tbody id="metrics-body"></tbody>
</table>
</div>

<h2>Scheduler Time-Series</h2>
<div class="trial-selector" id="trial-selector"></div>
<div class="charts-grid">
  <div class="card"><h3>Request Arrivals (cumulative)</h3><canvas id="chart-arrivals"></canvas></div>
  <div class="card"><h3>Queue Depth (Waiting vs Running)</h3><canvas id="chart-queue"></canvas></div>
  <div class="card"><h3>KV Cache Utilization (%)</h3><canvas id="chart-kv"></canvas></div>
  <div class="card"><h3>Preemptions (per step)</h3><canvas id="chart-preempt"></canvas></div>
  <div class="card"><h3>Step Latency (ms)</h3><canvas id="chart-step"></canvas></div>
  <div class="card"><h3>Batch Size</h3><canvas id="chart-batch"></canvas></div>
  <div class="card"><h3>Decode Padding Waste (%)</h3><canvas id="chart-padding"></canvas></div>
  <div class="card"><h3>GPU Busy % (smoothed)</h3><canvas id="chart-gpu"></canvas></div>
</div>
</div>

<div id="sweep-section" style="display:none">
  <h2 style="margin-top:36px">Saturation Sweep</h2>
  <div class="subtitle" id="sweep-subtitle"></div>
  <div class="charts-grid" style="margin-top:16px">
    <div class="card"><h3>Throughput vs Arrival Rate</h3><canvas id="chart-sweep-tput"></canvas></div>
    <div class="card"><h3>Latency vs Arrival Rate</h3><canvas id="chart-sweep-lat"></canvas></div>
  </div>
  <div class="table-wrap" style="margin-top:8px">
  <table id="sweep-table">
    <thead><tr>
      <th>Arrival Rate<br>(req/s)</th>
      <th>Throughput<br>(tok/s)</th>
      <th>TTFT p50<br>(ms)</th>
      <th>TTFT p99<br>(ms)</th>
      <th>E2E avg<br>(ms)</th>
      <th>Completed<br>(%)</th>
    </tr></thead>
    <tbody id="sweep-body"></tbody>
  </table>
  </div>
</div>

<script>
const BENCH = __BENCH_DATA__;
const DEBUG = __DEBUG_DATA__;
const SWEEP = __SWEEP_DATA__;
const SWEEP_ONLY = __SWEEP_ONLY__;

if (SWEEP_ONLY) {
  document.getElementById("bench-section").style.display = "none";
  document.getElementById("sweep-section").style.display = "";
  document.getElementById("run-ts").textContent =
    (SWEEP && SWEEP.timestamp) ? `sweep_${SWEEP.timestamp}` : "sweep";
}

// Shared chart colors (used by bench + sweep sections)
const BLUE  = "rgba(59,130,246,0.85)";
const GREEN = "rgba(34,197,94,0.85)";
const AMB   = "rgba(251,191,36,0.85)";
const PURP  = "rgba(168,85,247,0.85)";
const RED   = "rgba(239,68,68,0.85)";
const CYAN  = "rgba(34,211,238,0.85)";

// --- Populate timestamp ---
if (!SWEEP_ONLY) {
  document.getElementById("run-ts").textContent = BENCH.timestamp || "unknown";
}

// --- Populate params bar ---
(function() {
  const p = BENCH.params || {};
  const bar = document.getElementById("params-bar");
  if (!Object.keys(p).length) { bar.style.display = "none"; return; }
  const labels = {
    preset:        "Preset",
    description:   "Workload",
    model:         "Model",
    gpu:           "GPU",
    max_model_len: "Max context",
    arrival_rate:  "Arrival rate",
    decay_rate:    "Decay rate",
    duration_s:    "Duration (s)",
    input_min:     "Input min",
    input_max:     "Input max",
    output_min:    "Output min",
    output_max:    "Output max",
    max_input_len: "Max input tok",
    max_output_len:"Max output tok",
    num_trials:    "Trials",
    enforce_eager: "Eager mode",
  };
  for (const [k, lbl] of Object.entries(labels)) {
    if (!(k in p)) continue;
    let val = p[k];
    if (k === "model") val = val.split("/").pop();
    if (typeof val === "boolean") val = val ? "yes" : "no";
    const div = document.createElement("div");
    div.className = "param";
    div.innerHTML = `${lbl}:<span>${val}</span>`;
    bar.appendChild(div);
  }
})();

// --- Split DEBUG events by trial ---
function buildTrialEvents(debugEvents) {
  const trials = {};
  let current = null;
  for (const ev of debugEvents) {
    if (ev.event === "trial_start") {
      current = ev.trial;
      trials[current] = [];
    } else if (ev.event === "trial_end") {
      current = null;
    } else if (current !== null) {
      if (!trials[current]) trials[current] = [];
      trials[current].push(ev);
    }
  }
  return trials;
}

const trialEvents = buildTrialEvents(DEBUG);
const trialKeys = Object.keys(trialEvents).map(Number).sort((a,b)=>a-b);
const realTrialKeys = trialKeys.filter(k => k > 0);

// --- Metrics table ---
function ms(v)  { return v != null ? (v * 1000).toFixed(1) : "—"; }
function toks(v){ return v != null ? v.toFixed(1) : "—"; }

const tbody = document.getElementById("metrics-body");
function addRow(label, m, cls) {
  const tr = document.createElement("tr");
  if (cls) tr.className = cls;
  tr.innerHTML = `
    <td>${label}</td>
    <td>${toks(m.throughput)}</td>
    <td>${ms(m.p50_ttft)}</td>
    <td>${ms(m.p99_ttft)}</td>
    <td>${ms(m.avg_tpot)}</td>
    <td>${ms(m.avg_e2e)}</td>`;
  tbody.appendChild(tr);
}

addRow("Warmup", BENCH.warmup, "warmup");
BENCH.trials.forEach((m, i) => addRow(i + 1, m, ""));

const avg = {};
["throughput","p50_ttft","p99_ttft","avg_tpot","avg_e2e"].forEach(k => {
  avg[k] = BENCH.trials.reduce((s, m) => s + (m[k] || 0), 0) / BENCH.trials.length;
});
addRow("AVG", avg, "avg");

// --- Trial selector ---
const selector = document.getElementById("trial-selector");
let activeCharts = [];
let activeTrial = realTrialKeys[0] ?? trialKeys[0];

function destroyCharts() {
  activeCharts.forEach(c => c.destroy());
  activeCharts = [];
}

// CUDA graph batch size breakpoints (mirrors model_runner.py)
// graph_bs = [1, 2, 4, 8] + range(16, max_bs+1, 16); max_bs = min(max_num_seqs, 512)
const GRAPH_BS = [1, 2, 4, 8, ...Array.from({length: 32}, (_, i) => (i + 1) * 16)];

function inferGraphBs(numSeqs) {
  return GRAPH_BS.find(b => b >= numSeqs) ?? numSeqs;
}

function paddingWastePct(numSeqs, isPrefill) {
  if (isPrefill || numSeqs > 512) return 0;
  const gbs = inferGraphBs(numSeqs);
  return ((gbs - numSeqs) / gbs * 100);
}

function buildCharts(trialIdx) {
  destroyCharts();
  const allEvts  = trialEvents[trialIdx] || [];
  const schedEvts = allEvts.filter(e => e.event === "schedule");

  const ALL_IDS = ["chart-arrivals","chart-queue","chart-kv","chart-preempt","chart-step",
                   "chart-batch","chart-padding","chart-gpu"];

  if (!schedEvts.length) {
    ALL_IDS.forEach(id => {
      const c = new Chart(document.getElementById(id), {
        type:"line", data:{datasets:[]},
        options:{ plugins:{legend:{display:false}},
          scales:{ x:{display:false}, y:{display:false} } }
      });
      activeCharts.push(c);
    });
    return;
  }

  const ts0    = schedEvts[0].ts;
  const labels = schedEvts.map(e => ((e.ts - ts0) * 1000).toFixed(0));

  const waiting  = schedEvts.map(e => e.queues?.waiting ?? 0);
  const running  = schedEvts.map(e => e.queues?.running ?? 0);
  const kvUtil   = schedEvts.map(e => {
    const t = e.kv_blocks?.total ?? 1;
    const u = e.kv_blocks?.used  ?? 0;
    return +(u / t * 100).toFixed(1);
  });
  const batchSz  = schedEvts.map(e => e.num_seqs ?? 0);
  const stepLat  = schedEvts.map((e, i) =>
    i === 0 ? 0 : +((e.ts - schedEvts[i-1].ts) * 1000).toFixed(2)
  );
  const padding  = schedEvts.map(e =>
    +paddingWastePct(e.num_seqs ?? 0, e.is_prefill ?? false).toFixed(1)
  );

  // --- Cumulative request arrivals (sampled onto schedule event timeline) ---
  const arriveEvts = allEvts.filter(e => e.event === "request_arrive");
  const ts0sched   = schedEvts[0].ts;
  const arrivalCumulative = schedEvts.map(sched => {
    return arriveEvts.filter(a => a.ts <= sched.ts).length;
  });

  // --- Preemptions per step window (between consecutive schedule events) ---
  const preemptEvts = allEvts.filter(e => e.event === "preempt");
  const preemptPerStep = schedEvts.map((sched, i) => {
    const prevTs = i === 0 ? -Infinity : schedEvts[i - 1].ts;
    return preemptEvts.filter(p => p.ts > prevTs && p.ts <= sched.ts).length;
  });

  // --- Per-step GPU busy % (forward_ms / step_ms * 100) ---
  const preEvts  = allEvts.filter(e => e.event === "pre_forward");
  const postEvts = allEvts.filter(e => e.event === "post_forward");
  const gpuBusyRaw = schedEvts.map((sched, i) => {
    if (i === 0) return null;
    const prevTs = schedEvts[i - 1].ts;
    const pre  = preEvts.find(e  => e.ts >= prevTs && e.ts < sched.ts);
    const post = postEvts.find(e => e.ts >= prevTs && e.ts < sched.ts);
    if (!pre || !post) return null;
    const fwdMs  = (post.ts - pre.ts) * 1000;
    const stepMs = (sched.ts - prevTs) * 1000;
    return stepMs > 0 ? +(fwdMs / stepMs * 100).toFixed(1) : null;
  });

  // Rolling average smoother
  function rollingAvg(arr, win) {
    return arr.map((_, i) => {
      const half = Math.floor(win / 2);
      const slice = arr.slice(Math.max(0, i - half), Math.min(arr.length, i + half + 1))
                       .filter(v => v != null);
      return slice.length ? +(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(1) : null;
    });
  }
  const gpuBusySmoothed = rollingAvg(gpuBusyRaw, 15);

  const baseLineOpts = {
    responsive: true,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", boxWidth: 12 } } },
    scales: {
      x: { ticks: { color: "#64748b", maxTicksLimit: 8 },
           grid: { color: "#1e293b" },
           title: { display: true, text: "time (ms)", color: "#64748b" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "#334155" } },
    },
  };

  const pctScale = { ...baseLineOpts.scales.y, min: 0, max: 100,
                     ticks: { color: "#64748b", callback: v => v + "%" } };

  function lineDs(label, data, color, extra = {}) {
    return { label, data, borderColor: color,
             backgroundColor: color.replace("0.85","0.12"),
             borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.3, ...extra };
  }

  // Cumulative arrivals
  activeCharts.push(new Chart(document.getElementById("chart-arrivals"), {
    type: "line",
    data: { labels, datasets: [ lineDs("Cumulative arrivals", arrivalCumulative, CYAN) ] },
    options: baseLineOpts,
  }));

  // Queue depth
  activeCharts.push(new Chart(document.getElementById("chart-queue"), {
    type: "line",
    data: { labels, datasets: [
      lineDs("Waiting", waiting, BLUE),
      lineDs("Running", running, GREEN),
    ]},
    options: baseLineOpts,
  }));

  // KV utilization
  activeCharts.push(new Chart(document.getElementById("chart-kv"), {
    type: "line",
    data: { labels, datasets: [ lineDs("KV util %", kvUtil, AMB) ] },
    options: { ...baseLineOpts, scales: { ...baseLineOpts.scales, y: pctScale } },
  }));

  // Preemptions
  activeCharts.push(new Chart(document.getElementById("chart-preempt"), {
    type: "line",
    data: { labels, datasets: [
      lineDs("Preemptions", preemptPerStep, RED, { fill: true, pointRadius: 0 }),
    ]},
    options: { ...baseLineOpts,
      scales: { ...baseLineOpts.scales,
        y: { ...baseLineOpts.scales.y, min: 0, beginAtZero: true,
             title: { display: true, text: "count", color: "#64748b" } } },
    },
  }));

  // Step latency
  activeCharts.push(new Chart(document.getElementById("chart-step"), {
    type: "line",
    data: { labels, datasets: [ lineDs("Step ms", stepLat, PURP) ] },
    options: baseLineOpts,
  }));

  // Batch size
  activeCharts.push(new Chart(document.getElementById("chart-batch"), {
    type: "line",
    data: { labels, datasets: [ lineDs("Batch size", batchSz, GREEN) ] },
    options: baseLineOpts,
  }));

  // Padding waste (decode steps only; prefill = 0)
  activeCharts.push(new Chart(document.getElementById("chart-padding"), {
    type: "line",
    data: { labels, datasets: [ lineDs("Padding waste %", padding, RED) ] },
    options: { ...baseLineOpts, scales: { ...baseLineOpts.scales, y: pctScale } },
  }));

  // GPU Busy % (smoothed)
  activeCharts.push(new Chart(document.getElementById("chart-gpu"), {
    type: "line",
    data: { labels, datasets: [ lineDs("GPU busy % (smoothed)", gpuBusySmoothed, GREEN) ] },
    options: { ...baseLineOpts, scales: { ...baseLineOpts.scales, y: pctScale } },
  }));
}

// Build trial buttons (skip warmup = trial 0)
[...trialKeys].forEach(k => {
  const btn = document.createElement("button");
  btn.className = "trial-btn" + (k === activeTrial ? " active" : "");
  btn.textContent = k === 0 ? "Warmup" : `Trial ${k}`;
  btn.onclick = () => {
    activeTrial = k;
    document.querySelectorAll(".trial-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    buildCharts(k);
  };
  selector.appendChild(btn);
});

buildCharts(activeTrial);

// --- Sweep section ---
(function() {
  if (!SWEEP) return;

  // Support both multi-profile format (SWEEP.profiles) and legacy flat format (SWEEP.results)
  const profileGroups = SWEEP.profiles && SWEEP.profiles.length
    ? SWEEP.profiles
    : [{ profile: { label: "sweep", description: "" }, results: SWEEP.results || [] }];

  if (!profileGroups.length) return;
  document.getElementById("sweep-section").style.display = "";

  const sp = SWEEP.params || {};
  const status = SWEEP.status ? ` | status: ${SWEEP.status}` : "";
  document.getElementById("sweep-subtitle").textContent =
    `Uniform Poisson arrivals | warmup ${sp.warmup_duration_s ?? "?"}s + measure ${sp.measure_duration_s ?? "?"}s per rate${status}`;

  // All unique rates across profiles (for a common x-axis)
  const allRates = [...new Set(
    profileGroups.flatMap(g => g.results.map(r => r.arrival_rate))
  )].sort((a, b) => a - b);
  const rateLabels = allRates.map(r => r + " req/s");

  const PROFILE_COLORS = [
    { line: GREEN, lat50: "rgba(34,211,238,0.85)", lat99: "rgba(251,191,36,0.85)", e2e: "rgba(239,68,68,0.85)" },
    { line: BLUE,  lat50: "rgba(147,197,253,0.85)", lat99: "rgba(253,186,116,0.85)", e2e: "rgba(252,165,165,0.85)" },
    { line: PURP,  lat50: "rgba(196,181,253,0.85)", lat99: "rgba(253,230,138,0.85)", e2e: "rgba(254,202,202,0.85)" },
  ];

  const sweepBaseOpts = {
    responsive: true,
    animation: false,
    plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", boxWidth: 12 } } },
    scales: {
      x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" },
           title: { display: true, text: "arrival rate (req/s)", color: "#64748b" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "#334155" } },
    },
  };

  function lookupRate(results, rate, field) {
    const r = results.find(x => x.arrival_rate === rate);
    if (!r || (r.saturated && !("throughput" in r))) return null;
    const v = r[field];
    return v != null ? v : null;
  }

  // --- Throughput chart: one line per profile ---
  new Chart(document.getElementById("chart-sweep-tput"), {
    type: "line",
    data: {
      labels: rateLabels,
      datasets: profileGroups.map((g, i) => {
        const col = PROFILE_COLORS[i % PROFILE_COLORS.length];
        return {
          label: g.profile.label,
          data: allRates.map(r => {
            const v = lookupRate(g.results, r, "throughput");
            return v != null ? +v.toFixed(1) : null;
          }),
          borderColor: col.line,
          backgroundColor: col.line.replace("0.85","0.12"),
          borderWidth: 2, pointRadius: 4, fill: false, tension: 0.2,
          spanGaps: false,
        };
      }),
    },
    options: sweepBaseOpts,
  });

  // --- Latency chart: TTFT p99 per profile (clearest saturation signal) ---
  new Chart(document.getElementById("chart-sweep-lat"), {
    type: "line",
    data: {
      labels: rateLabels,
      datasets: profileGroups.flatMap((g, i) => {
        const col = PROFILE_COLORS[i % PROFILE_COLORS.length];
        return [
          {
            label: `${g.profile.label} — TTFT p99`,
            data: allRates.map(r => {
              const v = lookupRate(g.results, r, "p99_ttft");
              return v != null ? +(v * 1000).toFixed(1) : null;
            }),
            borderColor: col.lat99, backgroundColor: "transparent",
            borderWidth: 2, pointRadius: 4, fill: false, tension: 0.2,
            borderDash: [],
            spanGaps: false,
          },
          {
            label: `${g.profile.label} — E2E avg`,
            data: allRates.map(r => {
              const v = lookupRate(g.results, r, "avg_e2e");
              return v != null ? +(v * 1000).toFixed(1) : null;
            }),
            borderColor: col.e2e, backgroundColor: "transparent",
            borderWidth: 1.5, pointRadius: 3, fill: false, tension: 0.2,
            borderDash: [4, 3],
            spanGaps: false,
          },
        ];
      }),
    },
    options: { ...sweepBaseOpts,
      scales: { ...sweepBaseOpts.scales,
        y: { ...sweepBaseOpts.scales.y,
             title: { display: true, text: "latency (ms)", color: "#64748b" } } },
    },
  });

  // --- Per-profile tables ---
  const sbody = document.getElementById("sweep-body");

  profileGroups.forEach((g, gi) => {
    // Profile header row
    const hdr = document.createElement("tr");
    hdr.innerHTML = `<td colspan="7" style="background:#0f172a;color:#7dd3fc;font-weight:700;padding:10px 14px">
      ${g.profile.label} — ${g.profile.description || ""}
    </td>`;
    sbody.appendChild(hdr);

    g.results.forEach(r => {
      const tr = document.createElement("tr");
      if (r.saturated && !("throughput" in r)) {
        tr.style.color = "#64748b";
        tr.innerHTML = `<td>${r.arrival_rate}</td><td colspan="5" style="text-align:left">skipped (saturated)</td>`;
      } else {
        const pct = r.num_submitted ? (r.num_requests / r.num_submitted * 100).toFixed(0) + "%" : "—";
        if (r.saturated) tr.style.color = "#f87171";
        tr.innerHTML = `
          <td>${r.arrival_rate}</td>
          <td>${r.throughput?.toFixed(1) ?? "—"}</td>
          <td>${r.p50_ttft != null ? (r.p50_ttft*1000).toFixed(0) : "—"}</td>
          <td>${r.p99_ttft != null ? (r.p99_ttft*1000).toFixed(0) : "—"}</td>
          <td>${r.avg_e2e  != null ? (r.avg_e2e *1000).toFixed(0) : "—"}</td>
          <td>${pct}</td>`;
      }
      sbody.appendChild(tr);
    });
  });
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def build_page(bench_data: dict, debug_events: list[dict], sweep_data: dict | None,
               *, sweep_only: bool = False) -> str:
    bench_json = json.dumps(bench_data, indent=None)
    debug_json = json.dumps(debug_events, indent=None)
    sweep_json = json.dumps(sweep_data, indent=None) if sweep_data else "null"
    html = HTML_TEMPLATE.replace("__BENCH_DATA__", bench_json)
    html = html.replace("__DEBUG_DATA__", debug_json)
    html = html.replace("__SWEEP_DATA__", sweep_json)
    html = html.replace("__SWEEP_ONLY__", "true" if sweep_only else "false")
    return html


def serve(html: str, port: int):
    encoded = html.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt, *args):
            pass  # suppress request logs

    server = HTTPServer(("localhost", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\nDashboard running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nano-vllm benchmark dashboard")
    parser.add_argument("--timestamp", default=None,
                        help="Run timestamp (YYYYMMDD_HHMMSS). For --sweep, selects sweep_TIMESTAMP.json.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep-only mode: load sweep_*.json (no bench/debug required).")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.sweep:
        sweep_data = fetch_sweep_only(args.timestamp)
        bench_data = {
            "timestamp": sweep_data.get("timestamp", "unknown"),
            "warmup": {},
            "trials": [],
            "params": sweep_data.get("params", {}),
        }
        debug_events = []
        html = build_page(bench_data, debug_events, sweep_data, sweep_only=True)
    else:
        bench_data, debug_events, sweep_data = fetch_volume_files(args.timestamp)
        html = build_page(bench_data, debug_events, sweep_data)

    serve(html, args.port)


if __name__ == "__main__":
    main()
