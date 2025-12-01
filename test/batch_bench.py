import modal

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
volume = modal.Volume.from_name("nano-vllm-models", create_if_missing=False)

@app.function(gpu="A10", volumes={MODEL_STORE_PATH: volume})
def main():
    import os
    import time
    from random import randint, seed
    from nanovllm import LLM, SamplingParams
    

    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = "/vol/models/Qwen_Qwen3-0.6B"
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
