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

# Create a volume for storing models
volume = modal.Volume.from_name("nano-vllm-models", create_if_missing=True)
MODEL_STORE_PATH = "/vol/models"

@app.function(volumes={MODEL_STORE_PATH: volume})
def download_model(model_id: str = "Qwen/Qwen3-0.6B"):
    """Download and store model in the volume once"""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Install the package at runtime
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/root"], check=True)
    
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    
    local_model_path = Path(MODEL_STORE_PATH) / model_id.replace("/", "_")
    
    # Check if model already exists
    if local_model_path.exists() and (local_model_path / "config.json").exists():
        print(f"✅ Model {model_id} already exists at {local_model_path}")
        return str(local_model_path)
    
    print(f"📥 Downloading model {model_id} to {local_model_path}...")
    local_model_path.mkdir(parents=True, exist_ok=True)
    
    # Download tokenizer and model files to volume
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_model_path)
    
    config = AutoConfig.from_pretrained(model_id)
    
    # Force fp16 for FlashAttention compatibility
    import torch
    config.torch_dtype = torch.float16
    config.save_pretrained(local_model_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.save_pretrained(local_model_path)
    
    print(f"✅ Model downloaded to {local_model_path}")
    
    # Commit changes to volume
    volume.commit()
    print(f"✅ Model committed to volume")
    
    return str(local_model_path)

@app.function(gpu="A10", volumes={MODEL_STORE_PATH: volume})
def run_inference(model_id: str = "Qwen/Qwen3-0.6B"):
    """Run inference using model from volume"""
    import subprocess
    import sys
    from pathlib import Path
    
    # Install the package at runtime
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/root"], check=True)
    
    # Import inside the function (runs in cloud)
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    local_model_path = Path(MODEL_STORE_PATH) / model_id.replace("/", "_")
    
    # Try to load model from volume
    try:
        if not local_model_path.exists() or not (local_model_path / "config.json").exists():
            raise FileNotFoundError("Model not found")
        print(f"✅ Loading model from volume: {local_model_path}")
    except FileNotFoundError:
        print(f"⚠️ Model not found in volume, reloading volume...")
        volume.reload()  # Fetch latest changes from other containers
        if not local_model_path.exists() or not (local_model_path / "config.json").exists():
            raise FileNotFoundError(f"Model {model_id} not found in volume. Please run download_model first.")
    
    # Load model from volume (config now set to fp16 for FlashAttention)
    llm = LLM(str(local_model_path), enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # Load tokenizer from volume path  
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for prompt, output in zip(prompts, outputs):
        result = f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}"
        print(result)
        results.append(result)
    
    return "\n".join(results)


@app.local_entrypoint()
def main():
    """Download model and run inference"""
    model_id = "Qwen/Qwen3-0.6B"  # Use Qwen3 which matches your nano-vllm implementation
    
    # Step 1: Download model to volume (only needed once)
    print("📥 Ensuring model is downloaded to volume...")
    model_path = download_model.remote(model_id)
    print(f"✅ Model available at: {model_path}")
    
    # Step 2: Run inference using model from volume
    print("🚀 Running inference with GPU...")
    result = run_inference.remote(model_id)
    print(result)
    
    return result

if __name__ == "__main__":
    # Run the workflow
    main()
