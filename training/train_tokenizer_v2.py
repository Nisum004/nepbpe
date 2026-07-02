import sentencepiece as spm
import os
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
import tiktoken
import json

os.makedirs("tokenizer/nepbpe_spm", exist_ok=True)

corpus_file = "data/processed/nepali_clean.txt"

print("🔤 Training SentencePiece Nepali Tokenizer...")
print("(This works at Unicode character level — better for Devanagari)\n")

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix="tokenizer/nepbpe_spm/nepbpe",
    vocab_size=64000,
    character_coverage=0.9995,   # cover almost all Nepali characters
    model_type="bpe",            # BPE algorithm
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    input_sentence_size=500000,  # use up to 500k sentences
    shuffle_input_sentence=True,
    num_threads=8,
)

print("✅ SentencePiece model trained!")

# Load and test
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")

# Load GPT-4o tokenizer for comparison
gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")

test_sentences = [
    "तपाईंलाई कस्तो छ",
    "नेपाली भाषा धेरै सुन्दर छ",
    "काठमाडौं नेपालको राजधानी हो",
    "आज मौसम राम्रो छ",
    "नेपाल सरकारले नयाँ नीति लागू गर्ने निर्णय गरेको छ",
    "प्रधानमन्त्रीले आज संसदमा महत्त्वपूर्ण भाषण दिनुभयो",
    "नागरिकता प्रमाणपत्र प्राप्त गर्न आवश्यक कागजातहरू",
    "जग्गाको लालपुर्जा नामसारी गर्ने प्रक्रिया",
    "मलाई नेपाली खाना मन पर्छ",
    "पहाडको काखमा बसेको सानो गाउँमा एक किसानको परिवार थियो",
]

print("\n" + "="*70)
print("BENCHMARK: NepBPE-SPM vs GPT-4o")
print("="*70)

total_ours = 0
total_gpt4 = 0

for sentence in test_sentences:
    our_tokens = sp.encode_as_pieces(sentence)
    our_count = len(our_tokens)
    gpt4_count = len(gpt4_tokenizer.encode(sentence))

    total_ours += our_count
    total_gpt4 += gpt4_count

    saving = ((gpt4_count - our_count) / gpt4_count) * 100

    print(f"\n'{sentence}'")
    print(f"  NepBPE-SPM: {our_count:3d} tokens → {our_tokens[:6]}...")
    print(f"  GPT-4o:     {gpt4_count:3d} tokens")
    print(f"  Saving:     {saving:+.1f}%")

avg_ours = total_ours / len(test_sentences)
avg_gpt4 = total_gpt4 / len(test_sentences)
overall_saving = ((avg_gpt4 - avg_ours) / avg_gpt4) * 100

print("\n" + "="*70)
print("OVERALL RESULTS")
print("="*70)
print(f"\n  NepBPE-SPM avg: {avg_ours:.1f} tokens/sentence")
print(f"  GPT-4o avg:     {avg_gpt4:.1f} tokens/sentence")
print(f"  Saving vs GPT-4o: {overall_saving:.1f}%")

# Save results
results = {
    "model": "NepBPE-SPM (SentencePiece BPE)",
    "vocab_size": 32000,
    "avg_tokens_nepbpe_spm": avg_ours,
    "avg_tokens_gpt4": avg_gpt4,
    "saving_vs_gpt4_percent": overall_saving,
}

with open("benchmark/tokenizer_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ Results saved to benchmark/tokenizer_v2_results.json")
print("\n📌 Note: ByteLevelBPE was worse for Devanagari because each")
print("   character = 3 bytes = 3 starting units before any merging.")
print("   SentencePiece works at Unicode level: 1 character = 1 unit.")
print("   This difference is itself a key research finding.")