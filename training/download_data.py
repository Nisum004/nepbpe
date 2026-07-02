from datasets import load_dataset
import os

output_file = "data/raw/nepali_corpus.txt"

def append_dataset(dataset, text_field, source_name):
    print(f"💾 Appending {source_name} to corpus...")
    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            text = example[text_field].strip()
            if text and len(text) > 50:  # skip very short lines
                f.write(text + "\n")
                count += 1
            if i % 10000 == 0 and i > 0:
                print(f"  → {i:,} processed...")
    print(f"✅ Added {count:,} lines from {source_name}")

# Source 2 — OSCAR Nepali (Common Crawl based)
print("\n📥 Downloading OSCAR Nepali...")
try:
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="ne",
        split="train",
        trust_remote_code=True
    )
    append_dataset(dataset, "text", "OSCAR Nepali")
except Exception as e:
    print(f"⚠️ OSCAR failed: {e}")
    print("Trying alternative...")

# Source 3 — CulturaX Nepali (cleaned multilingual corpus)
print("\n📥 Downloading CulturaX Nepali...")
try:
    dataset = load_dataset(
        "uonlp/CulturaX",
        "ne",
        split="train",
        trust_remote_code=True
    )
    append_dataset(dataset, "text", "CulturaX Nepali")
except Exception as e:
    print(f"⚠️ CulturaX failed: {e}")

# Final stats
with open(output_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\n📊 Total lines: {len(lines):,}")
print(f"📦 Total file size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")