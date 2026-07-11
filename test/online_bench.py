import modal
import sys
import math
import random
import time
from statistics import median
from typing import List

# Constants for the test
NUM_SEQS = 256
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", 
        add_python="3.11"
    )
    .entrypoint([])  # removes chatty prints on entry
    .pip_install("ninja", "packaging", "wheel", "torch")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("transformers>=4.51.0", "xxhash")
    .apt_install(["git", "build-essential"])
    # Add local files LAST - Modal will install them at runtime
    .add_local_dir("nanovllm", "/root/nanovllm")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

app = modal.App("nano-vllm", image=image)
MODEL_STORE_PATH = "/vol/models"
LOGS_PATH = "/vol/logs"
model_volume = modal.Volume.from_name("nano-vllm-models", create_if_missing=False)
logs_volume = modal.Volume.from_name("nano-vllm-logs", create_if_missing=True)

@app.function(gpu="A10", volumes={MODEL_STORE_PATH: model_volume, LOGS_PATH: logs_volume})
async def load_test(num_trials: int = 5, debug: bool = True):
    import os
    import time
    import asyncio
    from random import randint, seed
    from nanovllm import LLM, SamplingParams
    
    # Enable internal nano-vllm debug prints inside the Modal container.
    # This is intentionally opt-in via the load_test() arg.
    if debug:
        from datetime import datetime
        os.environ["NANOVLLM_DEBUG"] = "1"
        # Set timestamp for this run so all logs use the same filename
        os.environ["NANOVLLM_DEBUG_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = "/vol/models/Qwen_Qwen3-0.6B"
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    throughputs = []
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}")
        throughput = await run_test(llm, arrival_rate=10.0, decay_rate=0.1, duration=10.0)
        throughputs.append(throughput)
        print(f"Trial {i+1} completed: {throughput:.2f} tok/s")

    # Proper shutdown
    await llm.shutdown()
    
    # Commit logs volume so debug logs persist
    logs_volume.commit()
    
    avg_throughput = sum(throughputs) / len(throughputs)
    print(f"\nAverage throughput: {avg_throughput:.2f} tok/s")
    print(f"   Median: {median(throughputs):.2f} tok/s")
    print(f"   Min: {min(throughputs):.2f} tok/s")
    print(f"   Max: {max(throughputs):.2f} tok/s")
    return avg_throughput

async def run_test(llm, arrival_rate: float = 10.0, decay_rate: float = 0.1, duration: float = 10.0):
    from random import randint, seed
    import asyncio
    from nanovllm import SamplingParams
    
    seed(0)


    interarrival_times = generate_interarrival_times(arrival_rate=arrival_rate, decay_rate=decay_rate, duration=duration)

    total_tokens = 0
    tasks = []
    start_time = time.time()
    
    for i in range(len(interarrival_times)):
        await asyncio.sleep(interarrival_times[i])
        prompt_token_ids = [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN))
        # Collect tasks instead of awaiting immediately
        # agenerate expects a list of prompts, so wrap single prompt in a list
        task = llm.agenerate([prompt_token_ids], sampling_params)
        tasks.append(task)
        total_tokens += sampling_params.max_tokens

    # Await all the tasks 
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time

    throughput = total_tokens / elapsed_time
    print(f"Total: {total_tokens}tok, Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s") 

    return throughput

def generate_interarrival_times(arrival_rate: float, decay_rate: float, duration: int) -> List[float]:
    """Generate interarrival times for a Poisson process"""
    interarrival_times = []
    time = 0.0
    while time<duration:
        current_arrival_rate = arrival_rate * math.exp(-decay_rate * time)
        interarrival_time = random.expovariate(current_arrival_rate)

        if time + interarrival_time >= duration:
            break

        time += interarrival_time
        interarrival_times.append(interarrival_time)

    return interarrival_times

if __name__ == "__main__":
    import asyncio

    with app.run():
        results = load_test.remote(num_trials=5)
