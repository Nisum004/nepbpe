from datasets import load_dataset
import os

output_file = "data/raw/nepali_corpus.txt"

print("📥 Downloading Nepali News dataset...")
try:
    dataset = load_dataset(
        "News-Media-Reliability/cc-news-nepali",
        split="train"
    )
    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text and len(text) > 50:
                f.write(text + "\n")
                count += 1
    print(f"✅ Added {count:,} lines")
except Exception as e:
    print(f"⚠️ Failed: {e}")

print("\n📥 Downloading IndicCorp Nepali...")
try:
    dataset = load_dataset(
        "ai4bharat/IndicNLP-dataset",
        "ne",
        split="train"
    )
    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text and len(text) > 50:
                f.write(text + "\n")
                count += 1
    print(f"✅ Added {count:,} lines")
except Exception as e:
    print(f"⚠️ Failed: {e}")

# Final stats
with open(output_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

size = os.path.getsize(output_file) / 1024 / 1024
print(f"\n📊 Total lines: {len(lines):,}")
print(f"📦 Total size: {size:.1f} MB")