from tokenizers import ByteLevelBPETokenizer
import os
import json

corpus_file = "data/processed/nepali_clean.txt"
output_dir = "tokenizer"
os.makedirs(output_dir, exist_ok=True)

print("🔤 Training Nepali BPE Tokenizer...")
print(f"📂 Corpus: {corpus_file}")

for vocab_size in [16000, 32000]:
    print(f"\n⚙️  Training with vocab_size={vocab_size}...")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=3,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    save_path = f"{output_dir}/nepbpe_{vocab_size}"
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_model(save_path)
    print(f"✅ Saved to {save_path}")

    # Quick test
    test_sentences = [
        "तपाईंलाई कस्तो छ",
        "नेपाली भाषा धेरै सुन्दर छ",
        "काठमाडौं नेपालको राजधानी हो",
        "आज मौसम राम्रो छ",
    ]

    print(f"\n📊 Sample tokenization (vocab={vocab_size}):")
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        print(f"  '{sentence}'")
        print(f"   → {encoded.tokens}")
        print(f"   → {len(encoded.tokens)} tokens")
        print()

print("✅ Tokenizer training complete!")