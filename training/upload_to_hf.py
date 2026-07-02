from huggingface_hub import HfApi, create_repo
import os

# Always run relative to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"📂 Working directory: {os.getcwd()}")

api    = HfApi()
username = "nisum04"

# Create repos
for repo_name in ["nepbpe-tokenizer", "nepbpe-184m"]:
    try:
        create_repo(f"{username}/{repo_name}", exist_ok=True)
        print(f"✅ Repo ready: {username}/{repo_name}")
    except Exception as e:
        print(f"⚠️ {e}")

# Upload tokenizer
print("\n📤 Uploading tokenizer...")
api.upload_folder(
    folder_path="tokenizer/nepbpe_spm",
    repo_id=f"{username}/nepbpe-tokenizer",
    repo_type="model",
)
print("✅ Tokenizer uploaded")

# Upload model weights
print("\n📤 Uploading model...")
api.upload_file(
    path_or_fileobj="checkpoints/best_model.npz",
    path_in_repo="best_model.npz",
    repo_id=f"{username}/nepbpe-184m",
    repo_type="model",
)

# Upload model config
import json
config = {
    "model_name": "NepBPE-184M",
    "parameters": 184063488,
    "architecture": "GPT-2 style transformer",
    "vocab_size": 64000,
    "context_len": 1024,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "tokenizer": "SentencePiece BPE",
    "language": "Nepali",
    "corpus": "Nepali Wikipedia",
    "token_reduction_vs_gpt4o": "83.9%",
    "token_reduction_vs_llama3": "71.5%",
}

with open("checkpoints/config.json", "w") as f:
    json.dump(config, f, indent=2)

api.upload_file(
    path_or_fileobj="checkpoints/config.json",
    path_in_repo="config.json",
    repo_id=f"{username}/nepbpe-184m",
    repo_type="model",
)

# Upload benchmark results
api.upload_file(
    path_or_fileobj="benchmark/final_results.json",
    path_in_repo="benchmark_results.json",
    repo_id=f"{username}/nepbpe-184m",
    repo_type="model",
)

print("✅ Model uploaded")
print(f"\n🌐 Tokenizer: https://huggingface.co/{username}/nepbpe-tokenizer")
print(f"🌐 Model:     https://huggingface.co/{username}/nepbpe-184m")