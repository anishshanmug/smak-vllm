import modal

# Use CUDA development image for flash-attn compilation (Modal's recommended approach)
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

@app.function()
def test_build():
    """Test that everything imported correctly"""
    import subprocess
    import sys
    
    # Install the package at runtime (flash-attn already in image)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/root"], check=True)
    
    # Test all imports
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import flash_attn  # Should work since it's in the image
    
    print(f"✅ Flash-attn version: {flash_attn.__version__}")
    print(f"✅ PyTorch version: {__import__('torch').__version__}")
    
    return "✅ All imports successful, including flash-attn!"

if __name__ == "__main__":
    # Test the build
    with app.run():
        result = test_build.remote()
        print(result)