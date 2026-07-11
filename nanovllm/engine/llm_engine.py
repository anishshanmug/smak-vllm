import atexit
import asyncio
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Dict, List, Optional
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        
        # Async engine state
        self._running = False
        self._engine_task: Optional[asyncio.Task] = None
        self._pending_requests: Dict[str, Dict] = {}  # request_id -> request_data
        self._sequence_to_request: Dict[int, str] = {}  # seq_id -> request_id
        self._loop = None
        
        atexit.register(self.exit)

    def _start_background_engine(self):
        """Start the continuous background processing loop"""
        if not self._running:
            self._running = True
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running, create one
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            self._engine_task = self._loop.create_task(self._engine_loop())
    
    async def _engine_loop(self):
        """Continuously process scheduler steps"""
        while self._running:
            if not self.scheduler.is_finished():
                #output, num_tokens = self.step()
                output, num_tokens = await asyncio.to_thread(self.step)
                self._handle_completed_sequences(output)
            else:
                await asyncio.sleep(0.001)  # Small delay when idle
    
    def _handle_completed_sequences(self, completed_outputs):
        """Route completed sequences to their original generate() calls"""
        for seq_id, token_ids, seq in completed_outputs:
            request_id = self._sequence_to_request.get(seq_id)
            if request_id and request_id in self._pending_requests:
                request_data = self._pending_requests[request_id]

                ttft = (seq.first_token_time - seq.submit_time) if seq.first_token_time and seq.submit_time else None
                e2e_latency = (seq.finish_time - seq.submit_time) if seq.finish_time and seq.submit_time else None

                # Store this sequence's result
                request_data["results"][seq_id] = {
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                    "ttft": ttft,
                    "e2e_latency": e2e_latency,
                    "num_output_tokens": len(token_ids),
                }
                
                # Check if this request is complete
                expected_count = len(request_data["sequence_ids"])
                actual_count = len(request_data["results"])
                
                if actual_count == expected_count:
                    # All sequences for this request are done
                    results = [request_data["results"][seq_id] 
                              for seq_id in request_data["sequence_ids"]]
                    request_data["future"].set_result(results)
                    
                    # Cleanup sequence tracking
                    for seq_id in request_data["sequence_ids"]:
                        if seq_id in self._sequence_to_request:
                            del self._sequence_to_request[seq_id]
    
    async def shutdown(self):
        """Gracefully shutdown the background engine"""
        atexit.unregister(self.exit)  # ← Prevent double cleanup
        self._running = False
        if self._engine_task:
            await self._engine_task
        self.exit()

    def exit(self):
        """Synchronous cleanup"""
        self._running = False
        if self._engine_task and not self._engine_task.done():
            self._engine_task.cancel()
        
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, request_id: str = None):
        """Add a single request, optionally part of a larger request group"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        
        # Track which request this sequence belongs to
        if request_id:
            self._sequence_to_request[seq.seq_id] = request_id
            
        self.scheduler.add(seq)
        return seq.seq_id
    
    def _submit_batch_request(self, prompts: List[str | List[int]], sampling_params: List[SamplingParams]) -> str:
        """Submit multiple prompts as one logical request"""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        # Add all prompts with same request_id
        sequence_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            seq_id = self.add_request(prompt, sp, request_id)
            sequence_ids.append(seq_id)
        
        self._pending_requests[request_id] = {
            "sequence_ids": sequence_ids,
            "future": future,
            "results": {}
        }
        
        return request_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """Synchronous generate - for backward compatibility"""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, _seq in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    
    async def agenerate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        """Async generate - returns only results for THIS call"""
        
        # Normalize inputs
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Start background engine if not running
        if not self._running:
            self._start_background_engine()
        
        # Submit request and get tracking ID
        request_id = self._submit_batch_request(prompts, sampling_params)
        
        # Wait for completion
        results = await self._pending_requests[request_id]["future"]
        
        # Cleanup and return
        del self._pending_requests[request_id]
        return results
