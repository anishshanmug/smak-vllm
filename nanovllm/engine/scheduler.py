import os
from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.debug_log import debug_log, is_debug_enabled


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,config.kv_capacity_threshold,config.chunk_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._debug = is_debug_enabled()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
        waiting_count = min(len(self.waiting), self.max_num_seqs)
        for _ in range(waiting_count):
            seq = self.waiting.popleft()
            remaining_prompt_tokens = seq.num_prompt_tokens - seq.num_chunked_tokens
            chunk_tokens = min(
                self.block_manager.chunk_size,
                remaining_prompt_tokens,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            if chunk_tokens <= 0 or not self.block_manager.can_allocate(seq, chunk_tokens):
                self.waiting.appendleft(seq)
                break

            seq.num_scheduled_tokens = chunk_tokens
            self.block_manager.allocate(seq, chunk_tokens)
            num_batched_tokens += chunk_tokens
            scheduled_seqs.append(seq)
            self.waiting.append(seq)
        if scheduled_seqs:
            if self._debug:
                dbg = {
                    "event": "schedule",
                    "is_prefill": True,
                    "num_seqs": len(scheduled_seqs),
                    "num_batched_tokens": num_batched_tokens,
                    "queues": {
                        "waiting": len(self.waiting),
                        "running": len(self.running),
                    },
                    "kv_blocks": {
                        "free": len(self.block_manager.free_block_ids),
                        "used": len(self.block_manager.used_block_ids),
                        "total": len(self.block_manager.blocks),
                    },
                }
                debug_log(dbg)
            return scheduled_seqs, True

        # decode
        num_seqs = 0
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
        
        if self._debug:
            dbg = {
                "event": "schedule",
                "is_prefill": False,
                "num_seqs": len(scheduled_seqs),
                "num_batched_tokens": len(scheduled_seqs),
                "queues": {
                    "waiting": len(self.waiting),
                    "running": len(self.running),
                },
                "kv_blocks": {
                    "free": len(self.block_manager.free_block_ids),
                    "used": len(self.block_manager.used_block_ids),
                    "total": len(self.block_manager.blocks),
                },
            }
            debug_log(dbg)
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.num_chunked_tokens = 0
        seq.num_scheduled_tokens = 0
        self.waiting.appendleft(seq)
        if self._debug:
            debug_log({
                "event": "preempt",
                "seq_id": seq.seq_id,
                "num_completion_tokens": seq.num_completion_tokens,
            })

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_scheduled_tokens:
                seq.num_chunked_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_chunked_tokens < seq.num_prompt_tokens:
                    continue
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            seq.append_token(token_id)
            if seq.num_completion_tokens == 1:
                seq.first_token_time = perf_counter()
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.finish_time = perf_counter()
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
