#!/usr/bin/env python3
"""
Visualize performance metrics from nano-vllm debug logs.

Usage:
    python scripts/visualize_logs.py logs/stage0_debug.log [--output-dir output/]
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required. Install it with: pip install matplotlib")
    exit(1)


def parse_log_file(log_path: Path) -> List[Dict[str, Any]]:
    """Parse JSON debug entries from log file."""
    events = []
    step = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[nanovllm-debug]"):
                continue
            
            # Extract JSON part
            json_str = line[len("[nanovllm-debug] "):]
            try:
                event = json.loads(json_str)
                event['step'] = step
                events.append(event)
                step += 1
            except json.JSONDecodeError:
                continue
    
    return events


def extract_schedule_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract schedule events with relevant metrics."""
    schedule_events = []
    for event in events:
        if event.get('event') == 'schedule':
            schedule_events.append(event)
    return schedule_events


def extract_forward_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract forward pass events with CUDA memory info."""
    forward_events = []
    for event in events:
        if event.get('event') in ['pre_forward', 'post_forward']:
            forward_events.append(event)
    return forward_events


def plot_batched_num_seqs(schedule_events: List[Dict[str, Any]], output_dir: Path, max_num_seqs: int = None):
    """Plot batched_num_seqs vs step with max_num_seqs limit."""
    steps = [e['step'] for e in schedule_events]
    num_seqs = [e.get('num_seqs', 0) for e in schedule_events]
    
    # Use provided limit or infer from data
    if max_num_seqs is None:
        max_num_seqs = max(num_seqs) if num_seqs else 256
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, num_seqs, label='batched_num_seqs', linewidth=1.5)
    if max_num_seqs:
        plt.axhline(y=max_num_seqs, color='r', linestyle='--', label=f'max_num_seqs limit ({max_num_seqs})')
    plt.xlabel('Step')
    plt.ylabel('Number of Sequences')
    plt.title('Active Concurrency vs Scheduler Ceiling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'batched_num_seqs.png', dpi=150)
    plt.close()


def plot_batched_tokens(schedule_events: List[Dict[str, Any]], output_dir: Path, max_num_batched_tokens: int = None):
    """Plot batched_tokens_to_compute vs step with max_num_batched_tokens limit."""
    steps = [e['step'] for e in schedule_events]
    tokens = [e.get('num_batched_tokens', 0) for e in schedule_events]
    
    # Use provided limit or infer from data
    if max_num_batched_tokens is None:
        max_num_batched_tokens = max(tokens) if tokens else 8192
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, tokens, label='batched_tokens_to_compute', linewidth=1.5)
    if max_num_batched_tokens:
        plt.axhline(y=max_num_batched_tokens, color='r', linestyle='--', label=f'max_num_batched_tokens limit ({max_num_batched_tokens})')
    plt.xlabel('Step')
    plt.ylabel('Tokens to Compute')
    plt.title('Effective Work per Step vs Token Batching Ceiling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'batched_tokens.png', dpi=150)
    plt.close()


def plot_kv_cache_saturation(schedule_events: List[Dict[str, Any]], output_dir: Path):
    """Plot KV cache saturation (used/total) vs step."""
    steps = [e['step'] for e in schedule_events]
    kv_blocks = [e.get('kv_blocks', {}) for e in schedule_events]
    used = [kv.get('used', 0) for kv in kv_blocks]
    total = [kv.get('total', 1) for kv in kv_blocks]  # Avoid division by zero
    
    saturation = [u / t if t > 0 else 0 for u, t in zip(used, total)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, saturation, label='KV Cache Saturation (used/total)', linewidth=1.5, color='orange')
    plt.axhline(y=1.0, color='r', linestyle='--', label='100% Capacity')
    plt.xlabel('Step')
    plt.ylabel('Saturation Ratio')
    plt.title('KV Cache Saturation vs Capacity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'kv_cache_saturation.png', dpi=150)
    plt.close()


def plot_cuda_memory(forward_events: List[Dict[str, Any]], output_dir: Path):
    """Plot CUDA memory metrics (alloc/reserved/free) vs step."""
    steps = [e['step'] for e in forward_events]
    cuda_data = [e.get('cuda', {}) for e in forward_events]
    
    alloc_gb = [c.get('alloc_gb', 0) for c in cuda_data]
    reserved_gb = [c.get('reserved_gb', 0) for c in cuda_data]
    free_gb = [c.get('free_gb', 0) for c in cuda_data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, alloc_gb, label='alloc_gb', linewidth=1.5)
    plt.plot(steps, reserved_gb, label='reserved_gb', linewidth=1.5)
    plt.plot(steps, free_gb, label='free_gb', linewidth=1.5)
    plt.xlabel('Step')
    plt.ylabel('Memory (GB)')
    plt.title('GPU Memory Pressure and Allocator Behavior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cuda_memory.png', dpi=150)
    plt.close()


def plot_avg_kv_footprint(schedule_events: List[Dict[str, Any]], output_dir: Path):
    """Plot average KV footprint per sequence (kv_blocks.used / batched_num_seqs) vs step."""
    steps = [e['step'] for e in schedule_events]
    kv_blocks = [e.get('kv_blocks', {}) for e in schedule_events]
    used = [kv.get('used', 0) for kv in kv_blocks]
    num_seqs = [e.get('num_seqs', 1) for e in schedule_events]  # Avoid division by zero
    
    avg_footprint = [u / n if n > 0 else 0 for u, n in zip(used, num_seqs)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, avg_footprint, label='Avg KV Blocks per Sequence', linewidth=1.5, color='purple')
    plt.xlabel('Step')
    plt.ylabel('KV Blocks per Sequence')
    plt.title('Average KV Footprint per Active Sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_kv_footprint.png', dpi=150)
    plt.close()


def plot_queue_sizes(schedule_events: List[Dict[str, Any]], output_dir: Path):
    """Plot queue sizes (running/waiting) vs step."""
    steps = [e['step'] for e in schedule_events]
    queues = [e.get('queues', {}) for e in schedule_events]
    
    running = [q.get('running', 0) for q in queues]
    waiting = [q.get('waiting', 0) for q in queues]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, running, label='running', linewidth=1.5, color='green')
    plt.plot(steps, waiting, label='waiting', linewidth=1.5, color='red')
    plt.xlabel('Step')
    plt.ylabel('Number of Sequences')
    plt.title('Backpressure and Admission Pressure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'queue_sizes.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize nano-vllm debug logs')
    parser.add_argument('log_file', type=Path, help='Path to debug log file')
    parser.add_argument('--output-dir', type=Path, default=Path('output'), 
                       help='Output directory for graphs (default: output/)')
    parser.add_argument('--max-num-seqs', type=int, default=None,
                       help='Maximum number of sequences limit (for reference line)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=None,
                       help='Maximum number of batched tokens limit (for reference line)')
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Create output directory based on log filename
    log_name = args.log_file.stem  # Get filename without extension
    output_dir = args.output_dir / log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing log file: {args.log_file}")
    print(f"Output directory: {output_dir}")
    events = parse_log_file(args.log_file)
    print(f"Found {len(events)} events")
    
    schedule_events = extract_schedule_events(events)
    forward_events = extract_forward_events(events)
    
    print(f"Found {len(schedule_events)} schedule events")
    print(f"Found {len(forward_events)} forward events")
    
    if not schedule_events:
        print("Warning: No schedule events found. Cannot generate most graphs.")
        return 1
    
    print("Generating graphs...")
    
    # Generate all plots
    plot_batched_num_seqs(schedule_events, output_dir, args.max_num_seqs)
    plot_batched_tokens(schedule_events, output_dir, args.max_num_batched_tokens)
    plot_kv_cache_saturation(schedule_events, output_dir)
    plot_avg_kv_footprint(schedule_events, output_dir)
    plot_queue_sizes(schedule_events, output_dir)
    
    if forward_events:
        plot_cuda_memory(forward_events, output_dir)
    else:
        print("Warning: No forward events found. Skipping CUDA memory plot.")
    
    print(f"Graphs saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())

