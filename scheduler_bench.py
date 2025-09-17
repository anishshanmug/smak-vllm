#!/usr/bin/env python3
"""
Nano-vLLM Scheduler Benchmark

A comprehensive benchmark suite for testing scheduler efficiency without requiring GPU dependencies.
This script wraps the actual scheduler classes and mocks GPU dependencies.

Usage: python scheduler_bench.py
"""

import time
import statistics
import sys
from random import randint, seed, choice
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from unittest.mock import MagicMock

# Mock GPU dependencies more thoroughly
def mock_gpu_dependencies():
    """Mock GPU-related modules to allow importing scheduler without GPU dependencies"""
    
    # Create a more complete triton mock
    class MockTriton:
        def __init__(self):
            self.__spec__ = type('MockSpec', (), {'origin': 'mock'})()
            self.__version__ = '2.0.0'
        
        def __getattr__(self, name):
            return MagicMock()
    
    # Mock triton completely
    sys.modules['triton'] = MockTriton()
    sys.modules['triton.language'] = MagicMock()
    sys.modules['triton.runtime'] = MagicMock()
    sys.modules['triton.ops'] = MagicMock()
    
    # Mock transformers if it tries to import triton
    def mock_transformers_import():
        original_import = __builtins__.__import__
        def patched_import(name, *args, **kwargs):
            if name.startswith('triton'):
                return sys.modules.get(name, MockTriton())
            return original_import(name, *args, **kwargs)
        __builtins__.__import__ = patched_import
    
    mock_transformers_import()

# Apply mocks before importing
mock_gpu_dependencies()

# Try to import directly from specific modules to avoid init cascades
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import specific modules directly
try:
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence, SequenceStatus  
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.sampling_params import SamplingParams
    IMPORTS_SUCCESSFUL = True
    print("✅ Successfully imported actual nanovllm scheduler classes")
except ImportError as e:
    print(f"⚠️  Could not import nanovllm modules ({e}). Using fallback implementations.")
    IMPORTS_SUCCESSFUL = False
    
    # Fallback implementations (using the previous standalone versions)
    from collections import deque
    from enum import Enum, auto
    from itertools import count
    from copy import copy
    
    # Simple hash fallback
    def simple_hash(token_ids: list[int], prefix: int = -1):
        if prefix != -1:
            return hash((prefix, tuple(token_ids)))
        else:
            return hash(tuple(token_ids))
    
    @dataclass 
    class SamplingParams:
        temperature: float = 1.0
        max_tokens: int = 16
        ignore_eos: bool = False

    class SequenceStatus(Enum):
        WAITING = auto()
        RUNNING = auto()
        FINISHED = auto()

    class Sequence:
        block_size = 256
        counter = count()

        def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
            self.seq_id = next(Sequence.counter)
            self.status = SequenceStatus.WAITING
            self.token_ids = copy(token_ids)
            self.last_token = token_ids[-1]
            self.num_tokens = len(self.token_ids)
            self.num_prompt_tokens = len(token_ids)
            self.num_cached_tokens = 0
            self.block_table = []
            self.temperature = sampling_params.temperature
            self.max_tokens = sampling_params.max_tokens
            self.ignore_eos = sampling_params.ignore_eos

        def __len__(self):
            return self.num_tokens

        def __getitem__(self, key):
            return self.token_ids[key]

        @property
        def is_finished(self):
            return self.status == SequenceStatus.FINISHED

        @property
        def num_completion_tokens(self):
            return self.num_tokens - self.num_prompt_tokens

        @property
        def num_blocks(self):
            return (self.num_tokens + self.block_size - 1) // self.block_size

        def block(self, i):
            assert 0 <= i < self.num_blocks
            return self.token_ids[i*self.block_size: (i+1)*self.block_size]

        def append_token(self, token_id: int):
            self.token_ids.append(token_id)
            self.last_token = token_id
            self.num_tokens += 1

    class Block:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_count = 0
            self.hash = -1
            self.token_ids = []

        def update(self, hash: int, token_ids: list[int]):
            self.hash = hash
            self.token_ids = token_ids

        def reset(self):
            self.ref_count = 1
            self.hash = -1
            self.token_ids = []

    class BlockManager:
        def __init__(self, num_blocks: int, block_size: int):
            self.block_size = block_size
            self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
            self.hash_to_block_id: dict[int, int] = dict()
            self.free_block_ids: deque[int] = deque(range(num_blocks))
            self.used_block_ids: set[int] = set()

        def compute_hash(self, token_ids: list[int], prefix: int = -1):
            return simple_hash(token_ids, prefix)

        def _allocate_block(self, block_id: int) -> Block:
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
            return self.blocks[block_id]

        def _deallocate_block(self, block_id: int) -> Block:
            assert self.blocks[block_id].ref_count == 0
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)

        def can_allocate(self, seq: Sequence) -> bool:
            return len(self.free_block_ids) >= seq.num_blocks

        def allocate(self, seq: Sequence):
            assert not seq.block_table
            for i in range(seq.num_blocks):
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                seq.block_table.append(block_id)

        def deallocate(self, seq: Sequence):
            for block_id in reversed(seq.block_table):
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            seq.num_cached_tokens = 0
            seq.block_table.clear()

        def can_append(self, seq: Sequence) -> bool:
            return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

        def may_append(self, seq: Sequence):
            if len(seq) % self.block_size == 1:
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                seq.block_table.append(block_id)

    class Scheduler:
        def __init__(self, config):
            self.max_num_seqs = config.max_num_seqs
            self.max_num_batched_tokens = config.max_num_batched_tokens
            self.eos = config.eos
            self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
            self.waiting: deque[Sequence] = deque()
            self.running: deque[Sequence] = deque()

        def is_finished(self):
            return not self.waiting and not self.running

        def add(self, seq: Sequence):
            self.waiting.append(seq)

        def schedule(self) -> tuple[list[Sequence], bool]:
            # prefill
            scheduled_seqs = []
            num_seqs = 0
            num_batched_tokens = 0
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                    break
                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
            if scheduled_seqs:
                return scheduled_seqs, True

            # decode
            while self.running and num_seqs < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
            assert scheduled_seqs
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

        def preempt(self, seq: Sequence):
            seq.status = SequenceStatus.WAITING
            self.block_manager.deallocate(seq)
            self.waiting.appendleft(seq)

        def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

# Create a simple test config that mimics the real Config interface
class TestConfig:
    def __init__(self, **kwargs):
        # Set defaults that match the real Config class
        self.model = kwargs.get('model', 'dummy')
        self.max_num_seqs = kwargs.get('max_num_seqs', 32)
        self.max_num_batched_tokens = kwargs.get('max_num_batched_tokens', 8192)
        self.eos = kwargs.get('eos', 2)
        self.kvcache_block_size = kwargs.get('kvcache_block_size', 256)
        self.num_kvcache_blocks = kwargs.get('num_kvcache_blocks', 1024)


@dataclass
class SchedulerMetrics:
    """Metrics for scheduler performance analysis"""
    total_scheduling_time: float = 0.0
    total_decisions: int = 0
    successful_prefills: int = 0
    successful_decodes: int = 0
    preemptions: int = 0
    sequences_completed: int = 0
    peak_memory_usage: int = 0
    average_batch_size: float = 0.0
    cache_hit_rate: float = 0.0
    
    @property
    def decisions_per_second(self) -> float:
        return self.total_decisions / self.total_scheduling_time if self.total_scheduling_time > 0 else 0
    
    @property
    def preemption_rate(self) -> float:
        return self.preemptions / self.total_decisions if self.total_decisions > 0 else 0


class MockSequenceGenerator:
    """Generates mock sequences for testing"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
    
    def create_sequence(self, prompt_len: int, max_tokens: int, temperature: float = 0.6) -> Sequence:
        """Create a mock sequence with random token IDs"""
        token_ids = [randint(0, self.vocab_size - 1) for _ in range(prompt_len)]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=True
        )
        return Sequence(token_ids, sampling_params)
    
    def create_workload(self, num_sequences: int, min_prompt: int = 50, max_prompt: int = 2048, 
                       min_output: int = 50, max_output: int = 1024) -> List[Sequence]:
        """Create a batch of sequences representing a workload"""
        sequences = []
        for _ in range(num_sequences):
            prompt_len = randint(min_prompt, max_prompt)
            max_tokens = randint(min_output, max_output)
            sequences.append(self.create_sequence(prompt_len, max_tokens))
        return sequences


class SchedulerBenchmark:
    """Comprehensive scheduler benchmark suite"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.generator = MockSequenceGenerator()
        
    def create_scheduler(self) -> Scheduler:
        """Create a fresh scheduler instance"""
        return Scheduler(self.config)
    
    def simulate_token_generation(self, sequences: List[Sequence]) -> List[int]:
        """Mock token generation - returns random tokens"""
        return [randint(0, 49999) for _ in sequences]
    
    def run_scheduling_cycles(self, scheduler: Scheduler, sequences: List[Sequence], 
                            max_steps: int = 1000) -> SchedulerMetrics:
        """Run scheduler cycles until completion or max steps"""
        metrics = SchedulerMetrics()
        
        # Add all sequences to scheduler
        for seq in sequences:
            scheduler.add(seq)
        
        step = 0
        batch_sizes = []
        
        while not scheduler.is_finished() and step < max_steps:
            step += 1
            
            # Measure scheduling time
            start_time = time.perf_counter()
            scheduled_seqs, is_prefill = scheduler.schedule()
            scheduling_time = time.perf_counter() - start_time
            
            metrics.total_scheduling_time += scheduling_time
            metrics.total_decisions += 1
            
            if scheduled_seqs:
                batch_sizes.append(len(scheduled_seqs))
                
                if is_prefill:
                    metrics.successful_prefills += 1
                else:
                    metrics.successful_decodes += 1
                
                # Simulate token generation and postprocessing
                token_ids = self.simulate_token_generation(scheduled_seqs)
                finished_before = sum(1 for seq in scheduled_seqs if seq.is_finished)
                scheduler.postprocess(scheduled_seqs, token_ids)
                finished_after = sum(1 for seq in scheduled_seqs if seq.is_finished)
                metrics.sequences_completed += (finished_after - finished_before)
        
        # Calculate final metrics
        if batch_sizes:
            metrics.average_batch_size = statistics.mean(batch_sizes)
        
        # Memory usage (approximate based on block manager state)
        metrics.peak_memory_usage = len(scheduler.block_manager.used_block_ids)
        
        return metrics
    
    def test_throughput(self, num_sequences: int = 100) -> Dict[str, float]:
        """Test scheduler throughput with various sequence configurations"""
        results = {}
        
        # Test small sequences
        small_seqs = self.generator.create_workload(num_sequences, 50, 200, 50, 200)
        scheduler = self.create_scheduler()
        metrics = self.run_scheduling_cycles(scheduler, small_seqs)
        results['small_sequences_dps'] = metrics.decisions_per_second
        
        # Test large sequences  
        large_seqs = self.generator.create_workload(num_sequences, 1000, 2000, 500, 1000)
        scheduler = self.create_scheduler()
        metrics = self.run_scheduling_cycles(scheduler, large_seqs)
        results['large_sequences_dps'] = metrics.decisions_per_second
        
        # Test mixed sequences
        mixed_seqs = (self.generator.create_workload(num_sequences//2, 50, 200, 50, 200) +
                     self.generator.create_workload(num_sequences//2, 1000, 2000, 500, 1000))
        scheduler = self.create_scheduler()
        metrics = self.run_scheduling_cycles(scheduler, mixed_seqs)
        results['mixed_sequences_dps'] = metrics.decisions_per_second
        
        return results
    
    def test_memory_pressure(self) -> Dict[str, float]:
        """Test scheduler behavior under memory pressure"""
        # Create more sequences than can fit in memory simultaneously
        sequences = self.generator.create_workload(200, 500, 1500, 200, 800)
        scheduler = self.create_scheduler()
        metrics = self.run_scheduling_cycles(scheduler, sequences)
        
        return {
            'preemption_rate': metrics.preemption_rate,
            'avg_batch_size_under_pressure': metrics.average_batch_size,
            'decisions_per_second': metrics.decisions_per_second,
            'peak_memory_blocks': metrics.peak_memory_usage
        }
    
    def test_burst_workload(self) -> Dict[str, float]:
        """Test scheduler with burst arrivals"""
        scheduler = self.create_scheduler()
        metrics = SchedulerMetrics()
        
        # Simulate bursts of requests
        for burst in range(5):
            # Add burst of sequences
            burst_seqs = self.generator.create_workload(50, 100, 500, 100, 300)
            for seq in burst_seqs:
                scheduler.add(seq)
            
            # Process some steps
            for _ in range(20):
                if scheduler.is_finished():
                    break
                    
                start_time = time.perf_counter()
                scheduled_seqs, is_prefill = scheduler.schedule()
                scheduling_time = time.perf_counter() - start_time
                
                metrics.total_scheduling_time += scheduling_time
                metrics.total_decisions += 1
                
                if scheduled_seqs:
                    token_ids = self.simulate_token_generation(scheduled_seqs)
                    scheduler.postprocess(scheduled_seqs, token_ids)
        
        return {
            'burst_decisions_per_second': metrics.decisions_per_second,
            'total_decisions': metrics.total_decisions
        }
    
    def test_fairness(self) -> Dict[str, float]:
        """Test scheduling fairness across different sequence sizes"""
        # Create sequences of different sizes
        small_seqs = self.generator.create_workload(50, 50, 100, 50, 100)
        large_seqs = self.generator.create_workload(50, 1000, 2000, 500, 1000)
        
        # Track completion times
        all_seqs = small_seqs + large_seqs
        scheduler = self.create_scheduler()
        
        for seq in all_seqs:
            scheduler.add(seq)
        
        completion_times = {}
        step = 0
        
        while not scheduler.is_finished() and step < 2000:
            step += 1
            scheduled_seqs, is_prefill = scheduler.schedule()
            
            if scheduled_seqs:
                token_ids = self.simulate_token_generation(scheduled_seqs)
                scheduler.postprocess(scheduled_seqs, token_ids)
                
                # Track when sequences finish
                for seq in scheduled_seqs:
                    if seq.is_finished and seq.seq_id not in completion_times:
                        completion_times[seq.seq_id] = step
        
        # Calculate fairness metrics
        small_completion_times = [completion_times.get(seq.seq_id, step) for seq in small_seqs]
        large_completion_times = [completion_times.get(seq.seq_id, step) for seq in large_seqs]
        
        return {
            'small_seq_avg_completion': statistics.mean(small_completion_times) if small_completion_times else 0,
            'large_seq_avg_completion': statistics.mean(large_completion_times) if large_completion_times else 0,
            'completion_ratio': (statistics.mean(large_completion_times) / statistics.mean(small_completion_times) 
                               if small_completion_times and large_completion_times else 0)
        }


def run_comprehensive_benchmark():
    """Run all scheduler benchmarks and report results"""
    print("🚀 Running Nano-vLLM Scheduler Benchmark")
    if IMPORTS_SUCCESSFUL:
        print("   Using ACTUAL nanovllm scheduler classes")
    else:
        print("   Using FALLBACK scheduler implementations")
    print("=" * 50)
    
    # Configure for testing
    config = TestConfig(
        model="dummy",
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        num_kvcache_blocks=1024,
        kvcache_block_size=256,
        eos=2  # Mock EOS token
    )
    
    benchmark = SchedulerBenchmark(config)
    
    # Test 1: Throughput
    print("\n📊 Testing Scheduler Throughput...")
    throughput_results = benchmark.test_throughput()
    for test_name, dps in throughput_results.items():
        print(f"  {test_name}: {dps:.2f} decisions/sec")
    
    # Test 2: Memory pressure
    print("\n🧠 Testing Memory Pressure Handling...")
    memory_results = benchmark.test_memory_pressure()
    for metric, value in memory_results.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test 3: Burst workload
    print("\n💥 Testing Burst Workload Handling...")
    burst_results = benchmark.test_burst_workload()
    for metric, value in burst_results.items():
        print(f"  {metric}: {value:.2f}")
    
    # Test 4: Fairness
    print("\n⚖️  Testing Scheduling Fairness...")
    fairness_results = benchmark.test_fairness()
    for metric, value in fairness_results.items():
        print(f"  {metric}: {value:.2f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 Summary:")
    avg_throughput = statistics.mean(throughput_results.values())
    print(f"  Average Throughput: {avg_throughput:.2f} decisions/sec")
    print(f"  Preemption Rate: {memory_results['preemption_rate']:.3f}")
    print(f"  Memory Efficiency: {memory_results['peak_memory_blocks']} blocks peak")
    
    # Efficiency score (higher is better)
    efficiency_score = (avg_throughput * (1 - memory_results['preemption_rate']) * 
                       memory_results['avg_batch_size_under_pressure'])
    print(f"  Scheduler Efficiency Score: {efficiency_score:.2f}")


if __name__ == "__main__":
    seed(42)  # For reproducible results
    run_comprehensive_benchmark()
