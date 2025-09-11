# Nano-vLLM Execution Flow: Deep Technical Analysis

This document provides a comprehensive technical analysis of what happens when you run `example.py` in the nano-vLLM codebase. Every component and interaction is traced with specific code citations.

## Table of Contents
1. [Overview](#overview)
2. [Phase 1: Initialization](#phase-1-initialization)
3. [Phase 2: Configuration and Setup](#phase-2-configuration-and-setup)
4. [Phase 3: Model Loading and Warmup](#phase-3-model-loading-and-warmup)
5. [Phase 4: Memory Management](#phase-4-memory-management)
6. [Phase 5: Prompt Processing](#phase-5-prompt-processing)
7. [Phase 6: Generation Loop](#phase-6-generation-loop)
8. [Phase 7: Attention and Model Execution](#phase-7-attention-and-model-execution)
9. [Phase 8: Sampling and Output](#phase-8-sampling-and-output)
10. [Performance Optimizations](#performance-optimizations)

## Overview

The execution begins with [`example.py`](example.py) which demonstrates a complete inference pipeline using nano-vLLM. The flow involves multiple sophisticated components working together to achieve high-performance language model inference.

```python
# example.py:6-9
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

## Phase 1: Initialization

### 1.1 LLM Class Creation

The `LLM` class is a simple wrapper around `LLMEngine`:

```python
# nanovllm/llm.py:4-5
class LLM(LLMEngine):
    pass
```

This delegates all functionality to the core [`LLMEngine`](nanovllm/engine/llm_engine.py) class.

### 1.2 LLMEngine Initialization

The core initialization happens in [`LLMEngine.__init__`](nanovllm/engine/llm_engine.py:17-34):

```python
# nanovllm/engine/llm_engine.py:17-21
def __init__(self, model, **kwargs):
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model, **config_kwargs)
```

This extracts configuration parameters and creates a [`Config`](nanovllm/config.py) object with validation.

## Phase 2: Configuration and Setup

### 2.1 Configuration Validation

The [`Config`](nanovllm/config.py:20-26) class performs critical validations:

```python
# nanovllm/config.py:20-26
def __post_init__(self):
    assert os.path.isdir(self.model)
    assert self.kvcache_block_size % 256 == 0
    assert 1 <= self.tensor_parallel_size <= 8
    self.hf_config = AutoConfig.from_pretrained(self.model)
    self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
    assert self.max_num_batched_tokens >= self.max_model_len
```

This ensures:
- Model directory exists
- KV-cache block size is aligned to 256 tokens
- Tensor parallelism is within bounds
- Model length constraints are respected

### 2.2 Process Management for Tensor Parallelism

Even with `tensor_parallel_size=1`, the engine sets up the multiprocessing infrastructure:

```python
# nanovllm/engine/llm_engine.py:22-29
self.ps = []
self.events = []
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
```

Since `tensor_parallel_size=1`, this loop doesn't execute, but the infrastructure is ready for scaling.

### 2.3 Core Component Initialization

```python
# nanovllm/engine/llm_engine.py:30-34
self.model_runner = ModelRunner(config, 0, self.events)
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
self.scheduler = Scheduler(config)
atexit.register(self.exit)
```

This creates the three core components:
- **ModelRunner**: Handles model execution and GPU operations
- **Tokenizer**: Converts text to/from tokens
- **Scheduler**: Manages sequence batching and memory allocation

## Phase 3: Model Loading and Warmup

### 3.1 ModelRunner Initialization

The [`ModelRunner.__init__`](nanovllm/engine/model_runner.py:17-48) method is the most complex initialization:

```python
# nanovllm/engine/model_runner.py:26-32
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(hf_config.torch_dtype)
torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

This sequence:
1. Initializes NCCL for distributed communication
2. Sets the CUDA device and data types
3. Creates the [`Qwen3ForCausalLM`](nanovllm/models/qwen3.py:180-211) model
4. Loads weights using [`load_model`](nanovllm/utils/loader.py:12-28)

### 3.2 Model Architecture

The [`Qwen3ForCausalLM`](nanovllm/models/qwen3.py:189-210) model consists of:

```python
# nanovllm/models/qwen3.py:194-197
self.model = Qwen3Model(config)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

The core [`Qwen3Model`](nanovllm/models/qwen3.py:162-177) contains:

```python
# nanovllm/models/qwen3.py:163-165
self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### 3.3 Weight Loading Process

The [`load_model`](nanovllm/utils/loader.py:12-28) function handles complex weight loading:

```python
# nanovllm/utils/loader.py:14-28
packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
for file in glob(os.path.join(path, "*.safetensors")):
    with safe_open(file, "pt", "cpu") as f:
        for weight_name in f.keys():
            for k in packed_modules_mapping:
                if k in weight_name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = weight_name.replace(k, v)
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                    break
            else:
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))
```

This handles:
- **Packed weight mapping**: Maps original parameter names to optimized packed structures
- **Safetensors loading**: Efficiently loads weights from disk
- **Custom weight loaders**: Allows parameters to define custom loading logic

### 3.4 Model Warmup

The [`warmup_model`](nanovllm/engine/model_runner.py:91-98) method optimizes memory allocation:

```python
# nanovllm/engine/model_runner.py:91-98
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

This runs a dummy forward pass to:
- Allocate all CUDA kernels and memory pools
- Measure peak memory usage for capacity planning
- Optimize GPU memory layout

## Phase 4: Memory Management

### 4.1 KV-Cache Allocation

The [`allocate_kv_cache`](nanovllm/engine/model_runner.py:100-117) method is crucial for performance:

```python
# nanovllm/engine/model_runner.py:103-111
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
num_kv_heads = hf_config.num_key_value_heads // self.world_size
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
assert config.num_kvcache_blocks > 0
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
```

This calculation:
1. **Measures available GPU memory** after model loading
2. **Calculates block memory requirements** for keys and values
3. **Determines optimal number of cache blocks** within memory budget
4. **Allocates the KV-cache tensor** with shape `[K/V, layers, blocks, block_size, heads, head_dim]`

### 4.2 Cache Assignment to Attention Layers

```python
# nanovllm/engine/model_runner.py:112-117
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

This assigns cache slices to each attention layer for efficient memory access.

### 4.3 Block Manager

The [`BlockManager`](nanovllm/engine/block_manager.py:26-113) handles advanced memory optimizations:

```python
# nanovllm/engine/block_manager.py:28-33
def __init__(self, num_blocks: int, block_size: int):
    self.block_size = block_size
    self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
    self.hash_to_block_id: dict[int, int] = dict()
    self.free_block_ids: deque[int] = deque(range(num_blocks))
    self.used_block_ids: set[int] = set()
```

Key features:
- **Block-based allocation**: Memory is managed in fixed-size blocks
- **Hash-based deduplication**: Identical token sequences share cache blocks
- **Reference counting**: Blocks can be shared across sequences

## Phase 5: Prompt Processing

### 5.1 Sampling Parameters

The [`SamplingParams`](nanovllm/sampling_params.py:4-11) class defines generation behavior:

```python
# nanovllm/sampling_params.py:10-11
def __post_init__(self):
    assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

This enforces probabilistic sampling (no greedy decoding) for diversity.

### 5.2 Chat Template Application

In [`example.py`](example.py:16-23), prompts are formatted for chat:

```python
# example.py:17-21
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
```

This converts raw text into the model's expected conversation format.

### 5.3 Sequence Creation

The [`generate`](nanovllm/engine/llm_engine.py:59-93) method processes prompts:

```python
# nanovllm/engine/llm_engine.py:69-70
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

Each call to [`add_request`](nanovllm/engine/llm_engine.py:42-46) creates a [`Sequence`](nanovllm/engine/sequence.py:14-84):

```python
# nanovllm/engine/sequence.py:18-29
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
```

## Phase 6: Generation Loop

### 6.1 Main Generation Loop

The core generation happens in [`generate`](nanovllm/engine/llm_engine.py:73-88):

```python
# nanovllm/engine/llm_engine.py:73-88
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
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids
        if use_tqdm:
            pbar.update(1)
```

### 6.2 Scheduling Logic

Each [`step`](nanovllm/engine/llm_engine.py:48-54) involves sophisticated scheduling via [`Scheduler.schedule`](nanovllm/engine/scheduler.py:24-58):

```python
# nanovllm/engine/scheduler.py:24-41
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
```

This implements **two-phase scheduling**:

1. **Prefill Phase**: Process new sequences (initial prompt processing)
   - Respects token budget (`max_num_batched_tokens`)
   - Checks memory availability via `BlockManager.can_allocate`
   - Allocates cache blocks for each sequence

2. **Decode Phase**: Continue existing sequences (token generation)
```python
# nanovllm/engine/scheduler.py:43-58
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
```

The decode phase includes **preemption logic** for memory pressure.

### 6.3 Memory Allocation with Prefix Caching

The [`BlockManager.allocate`](nanovllm/engine/block_manager.py:59-82) method implements sophisticated prefix caching:

```python
# nanovllm/engine/block_manager.py:59-82
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

This implements **automatic prefix caching**:
- Computes hashes for each block of tokens
- Reuses existing blocks when token sequences match
- Maintains reference counting for shared blocks
- Falls back to new allocation on cache misses

## Phase 7: Attention and Model Execution

### 7.1 Input Preparation

The [`ModelRunner.run`](nanovllm/engine/model_runner.py:207-213) method coordinates execution:

```python
# nanovllm/engine/model_runner.py:207-213
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

### 7.2 Prefill Preparation

The [`prepare_prefill`](nanovllm/engine/model_runner.py:125-161) method handles complex batching:

```python
# nanovllm/engine/model_runner.py:125-152
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    for seq in seqs:
        seqlen = len(seq)
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        # ... slot mapping logic
```

This creates:
- **Variable-length batching**: Different sequences can have different lengths
- **Cumulative sequence lengths**: For efficient FlashAttention batching
- **Position encodings**: Absolute positions for each token
- **Slot mapping**: Maps tokens to KV-cache locations

### 7.3 Context Management

The global context system ([`utils/context.py`](nanovllm/utils/context.py)) enables parameter passing:

```python
# nanovllm/utils/context.py:21-23
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
```

This avoids passing many parameters through the model forward pass.

### 7.4 Attention Computation

The [`Attention`](nanovllm/layers/attention.py:43-75) layer handles both prefill and decode:

```python
# nanovllm/layers/attention.py:59-75
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                   max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                   softmax_scale=self.scale, causal=True, block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                    softmax_scale=self.scale, causal=True)
    return o
```

**Key optimizations**:
- **KV-cache storage**: Uses custom Triton kernel for efficient cache updates
- **FlashAttention integration**: Uses memory-efficient attention algorithms
- **Prefix cache support**: Reuses cached attention for repeated prefixes
- **Variable-length attention**: Efficiently processes batches with different sequence lengths

### 7.5 Triton KV-Cache Kernel

The custom [`store_kvcache_kernel`](nanovllm/layers/attention.py:10-31) provides optimized cache updates:

```python
# nanovllm/layers/attention.py:10-31
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

This kernel efficiently stores computed keys/values into the paged cache structure.

## Phase 8: Sampling and Output

### 8.1 Sampling Implementation

The [`Sampler`](nanovllm/layers/sampler.py:5-15) implements temperature-based sampling:

```python
# nanovllm/layers/sampler.py:10-15
@torch.compile
def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

This implements **Gumbel-max sampling**:
1. Scale logits by temperature
2. Compute softmax probabilities  
3. Add Gumbel noise and select argmax
4. Uses `torch.compile` for optimization

### 8.2 Post-processing

The [`Scheduler.postprocess`](nanovllm/engine/scheduler.py:65-71) updates sequences:

```python
# nanovllm/engine/scheduler.py:65-71
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

This handles:
- **Token appending**: Adds new tokens to sequence
- **Stopping criteria**: Checks for EOS tokens or length limits
- **Memory cleanup**: Deallocates finished sequences

### 8.3 Output Formatting

Finally, [`generate`](nanovllm/engine/llm_engine.py:89-90) formats results:

```python
# nanovllm/engine/llm_engine.py:89-90
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

This creates the final output format with both decoded text and raw token IDs.

## Performance Optimizations

### CUDA Graph Capture (Disabled in Example)

When `enforce_eager=False`, the system captures CUDA graphs via [`capture_cudagraph`](nanovllm/engine/model_runner.py:215-250):

```python
# nanovllm/engine/model_runner.py:215-241
@torch.inference_mode()
def capture_cudagraph(self):
    # ... setup code ...
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
```

CUDA graphs eliminate kernel launch overhead for decode operations.

### Memory Efficiency Features

1. **Block-based KV-cache**: Reduces memory fragmentation
2. **Prefix caching**: Automatically shares computation for repeated prefixes
3. **Dynamic batching**: Maximizes GPU utilization
4. **Reference counting**: Efficient memory sharing
5. **FlashAttention**: Memory-efficient attention computation

### Throughput Monitoring

The system tracks two key metrics:
- **Prefill throughput**: Tokens processed per second during initial processing
- **Decode throughput**: Tokens generated per second during autoregressive generation

These metrics help identify bottlenecks and optimize performance.

## Conclusion

This execution flow demonstrates sophisticated engineering combining:
- **Advanced memory management** with block-based allocation and prefix caching
- **Optimized attention computation** using FlashAttention and custom kernels
- **Intelligent scheduling** with preemption and batching
- **Performance optimizations** including CUDA graphs and compilation
- **Robust error handling** and resource cleanup

The result is a high-performance inference engine that efficiently utilizes GPU resources while maintaining the flexibility to handle various workloads and scaling requirements.

## Anish Improvements

- run without GPU
- support other models
- add spec dec if using 1.5-3B target

- understand current processes and read SOTA improvements to implement

- scheduling + batching improvements:
    - priority queues over FIFO
    - Better bin packing (knapsack algorithm for batch filling)
    - paged attention? if not already implement
