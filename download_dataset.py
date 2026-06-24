import os
from huggingface_hub import snapshot_download

def download_nemotron_dataset():
    print("Initializing high-speed dataset download...")
    
    # Enable high-speed rust-based transfer for large LFS files
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    
    repo_id = "meet-the-1337/Nemotron-AIQ-Agentic-Safety-Dataset-1.0"
    local_dir = "./Nemotron-Dataset"
    
    print(f"Downloading 3.3GB dataset from {repo_id}...")
    print("This may take a few minutes depending on your internet connection.")
    
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            resume_download=True,
            max_workers=8  # Parallel downloading
        )
        print(f"\n✅ Success! Dataset fully downloaded to: {path}")
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("Please ensure you have internet connectivity and sufficient disk space.")

if __name__ == "__main__":
    # Ensure huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("Missing required package. Please run: pip install huggingface_hub hf_transfer")
        exit(1)
        
    download_nemotron_dataset()
