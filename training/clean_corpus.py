import re
import os
from tqdm import tqdm

input_file = "data/raw/nepali_corpus.txt"
output_file = "data/processed/nepali_clean.txt"

os.makedirs("data/processed", exist_ok=True)

# Nepali unicode range: \u0900-\u097F
def is_mostly_nepali(text, threshold=0.4):
    nepali_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return nepali_chars / len(text) >= threshold if text else False

def clean_line(text):
    text = text.strip()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove lines with too many special characters
    text = re.sub(r'[^\u0900-\u097F\s\u0964\u0965\.,!?०-९0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("🧹 Cleaning corpus...")

kept = 0
skipped = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    lines = fin.readlines()
    for line in tqdm(lines, desc="Cleaning"):
        line = line.strip()

        # Skip empty lines
        if not line:
            skipped += 1
            continue

        # Skip very short lines
        if len(line) < 30:
            skipped += 1
            continue

        # Skip lines that aren't mostly Nepali
        if not is_mostly_nepali(line):
            skipped += 1
            continue

        cleaned = clean_line(line)

        if len(cleaned) < 30:
            skipped += 1
            continue

        fout.write(cleaned + "\n")
        kept += 1

print(f"\n✅ Done!")
print(f"✅ Kept:    {kept:,} lines")
print(f"❌ Skipped: {skipped:,} lines")
size = os.path.getsize(output_file) / 1024 / 1024
print(f"📦 Clean corpus size: {size:.1f} MB")