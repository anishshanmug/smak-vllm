import os
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.debug_log import debug_log, is_debug_enabled


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._debug = is_debug_enabled()

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
            is_prefill = True
            if self._debug:
                dbg = {
                    "event": "schedule",
                    "is_prefill": is_prefill,
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
        
        is_prefill = False
        if self._debug:
            dbg = {
                "event": "schedule",
                "is_prefill": is_prefill,
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
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
