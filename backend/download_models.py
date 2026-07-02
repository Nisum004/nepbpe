from huggingface_hub import hf_hub_download
import os

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("tokenizer/nepbpe_spm", exist_ok=True)

print("Downloading best_model.pt...")
hf_hub_download(
    repo_id="nisum04/nepbpe-184m",
    filename="best_model.pt",
    local_dir="checkpoints"
)

print("Downloading chat_model.pt...")
hf_hub_download(
    repo_id="nisum04/nepbpe-184m",
    filename="chat_model.pt",
    local_dir="checkpoints"
)

print("Downloading tokenizer...")
hf_hub_download(
    repo_id="nisum04/nepbpe-tokenizer",
    filename="nepbpe.model",
    local_dir="tokenizer/nepbpe_spm"
)

print("✅ All files downloaded")