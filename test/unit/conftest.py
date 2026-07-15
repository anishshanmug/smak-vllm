from collections import Counter
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

# Import engine modules without executing nanovllm/__init__.py, which eagerly
# imports the GPU model runner and requires Triton even for CPU-only tests.
package = ModuleType("nanovllm")
package.__path__ = [str(Path(__file__).parents[2] / "nanovllm")]
sys.modules.setdefault("nanovllm", package)

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@pytest.fixture
def make_sequence():
    def factory(num_prompt_tokens: int, **sampling_overrides) -> Sequence:
        params = SamplingParams(
            temperature=sampling_overrides.pop("temperature", 1.0),
            max_tokens=sampling_overrides.pop("max_tokens", 4),
            ignore_eos=sampling_overrides.pop("ignore_eos", True),
        )
        if sampling_overrides:
            raise TypeError(f"Unknown sampling overrides: {sampling_overrides}")
        return Sequence(list(range(num_prompt_tokens)), params)

    return factory


@pytest.fixture
def make_scheduler():
    def factory(**overrides) -> Scheduler:
        values = {
            "max_num_seqs": 1,
            "max_num_batched_tokens": 512,
            "eos": -1,
            "num_kvcache_blocks": 8,
            "kvcache_block_size": 256,
            "kv_capacity_threshold": 1.0,
            "chunk_size": 512,
        }
        values.update(overrides)
        return Scheduler(SimpleNamespace(**values))

    return factory


@pytest.fixture
def assert_block_invariants():
    def check(scheduler: Scheduler) -> None:
        manager = scheduler.block_manager
        free_ids = list(manager.free_block_ids)
        free = set(free_ids)
        used = set(manager.used_block_ids)
        all_ids = set(range(len(manager.blocks)))

        assert len(free_ids) == len(free)
        assert free.isdisjoint(used)
        assert free | used == all_ids

        active_sequences = [*scheduler.waiting, *scheduler.running]
        ownership = Counter(
            block_id
            for seq in active_sequences
            for block_id in seq.block_table
        )
        for block in manager.blocks:
            assert block.ref_count == ownership[block.block_id]
            assert (block.block_id in used) == (block.ref_count > 0)

        for seq in active_sequences:
            assert len(seq.block_table) == len(set(seq.block_table))
            assert all(block_id in all_ids for block_id in seq.block_table)
            assert 0 <= seq.num_cached_tokens <= seq.num_chunked_tokens
            assert seq.num_chunked_tokens <= seq.num_prompt_tokens

    return check
