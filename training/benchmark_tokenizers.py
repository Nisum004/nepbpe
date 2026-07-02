import tiktoken
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import json

# Load our tokenizer
our_tokenizer = ByteLevelBPETokenizer(
    "tokenizer/nepbpe_32000/vocab.json",
    "tokenizer/nepbpe_32000/merges.txt"
)

# Load GPT-4o tokenizer
gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")

# Load Llama3 tokenizer
print("Loading Llama3 tokenizer...")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Test sentences across categories
test_sentences = {
    "Conversational": [
        "तपाईंलाई कस्तो छ",
        "आज मौसम राम्रो छ",
        "तपाईको नाम के हो",
        "मलाई नेपाली खाना मन पर्छ",
        "हामी साथीहरू मिलेर खेल्छौं",
    ],
    "News": [
        "काठमाडौं नेपालको राजधानी र सबैभन्दा ठूलो शहर हो",
        "नेपाल सरकारले नयाँ नीति लागू गर्ने निर्णय गरेको छ",
        "प्रधानमन्त्रीले आज संसदमा महत्त्वपूर्ण भाषण दिनुभयो",
        "देशको अर्थतन्त्र सुधार हुँदै गएको विज्ञहरूले बताए",
    ],
    "Government": [
        "नागरिकता प्रमाणपत्र प्राप्त गर्न आवश्यक कागजातहरू",
        "जग्गाको लालपुर्जा नामसारी गर्ने प्रक्रिया",
        "राहदानीको लागि आवेदन दिने तरिका",
    ],
    "Literary": [
        "नेपाली साहित्यमा कविता र गजलको विशेष स्थान छ",
        "पहाडको काखमा बसेको सानो गाउँमा एक किसानको परिवार थियो",
    ]
}

results = {}
print("\n" + "="*70)
print("TOKENIZER BENCHMARK: NepBPE vs GPT-4o vs Llama3")
print("="*70)

total_ours = 0
total_gpt4 = 0
total_llama = 0
total_sentences = 0

for category, sentences in test_sentences.items():
    print(f"\n📂 Category: {category}")
    print("-" * 50)
    results[category] = []

    for sentence in sentences:
        our_tokens = len(our_tokenizer.encode(sentence).tokens)
        gpt4_tokens = len(gpt4_tokenizer.encode(sentence))
        llama_tokens = len(llama_tokenizer.encode(sentence))

        total_ours += our_tokens
        total_gpt4 += gpt4_tokens
        total_llama += llama_tokens
        total_sentences += 1

        saving_vs_gpt4 = ((gpt4_tokens - our_tokens) / gpt4_tokens) * 100
        saving_vs_llama = ((llama_tokens - our_tokens) / llama_tokens) * 100

        results[category].append({
            "sentence": sentence,
            "our": our_tokens,
            "gpt4": gpt4_tokens,
            "llama": llama_tokens,
        })

        print(f"\n  '{sentence}'")
        print(f"  NepBPE:  {our_tokens:3d} tokens")
        print(f"  GPT-4o:  {gpt4_tokens:3d} tokens  ({saving_vs_gpt4:+.1f}% vs ours)")
        print(f"  Llama3:  {llama_tokens:3d} tokens  ({saving_vs_llama:+.1f}% vs ours)")

print("\n" + "="*70)
print("OVERALL RESULTS")
print("="*70)
avg_ours  = total_ours / total_sentences
avg_gpt4  = total_gpt4 / total_sentences
avg_llama = total_llama / total_sentences

print(f"\n  Average tokens per sentence:")
print(f"  NepBPE:  {avg_ours:.1f}")
print(f"  GPT-4o:  {avg_gpt4:.1f}")
print(f"  Llama3:  {avg_llama:.1f}")

saving_gpt4  = ((avg_gpt4 - avg_ours) / avg_gpt4) * 100
saving_llama = ((avg_llama - avg_ours) / avg_llama) * 100

print(f"\n  NepBPE uses {saving_gpt4:.1f}% fewer tokens than GPT-4o")
print(f"  NepBPE uses {saving_llama:.1f}% fewer tokens than Llama3")

# Cost comparison (GPT-4o input: $5 per 1M tokens)
cost_per_million = 5.0
our_cost  = (avg_ours / avg_gpt4) * cost_per_million
gpt4_cost = cost_per_million

print(f"\n  💰 Cost per 1M words of Nepali")