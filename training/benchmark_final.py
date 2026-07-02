import sentencepiece as spm
import tiktoken
from transformers import AutoTokenizer
import json
import os

os.makedirs("benchmark", exist_ok=True)

# Load our best tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")

# GPT-4o
gpt4 = tiktoken.get_encoding("cl100k_base")

# Llama3
print("Loading Llama3 tokenizer...")
llama3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Claude tokenizer (same as GPT-4o cl100k roughly, but let's use GPT-4 for now)
# We'll note in thesis that Claude uses similar BPE

test_sentences = {
    "Conversational": [
        "तपाईंलाई कस्तो छ",
        "आज मौसम राम्रो छ",
        "तपाईको नाम के हो",
        "मलाई नेपाली खाना मन पर्छ",
        "हामी साथीहरू मिलेर खेल्छौं",
        "भोलि तपाईं कहाँ जानुहुन्छ",
    ],
    "News": [
        "काठमाडौं नेपालको राजधानी र सबैभन्दा ठूलो शहर हो",
        "नेपाल सरकारले नयाँ नीति लागू गर्ने निर्णय गरेको छ",
        "प्रधानमन्त्रीले आज संसदमा महत्त्वपूर्ण भाषण दिनुभयो",
        "देशको अर्थतन्त्र सुधार हुँदै गएको विज्ञहरूले बताए",
        "नेपालमा पर्यटन उद्योग तीव्र गतिमा विकास भइरहेको छ",
    ],
    "Government": [
        "नागरिकता प्रमाणपत्र प्राप्त गर्न आवश्यक कागजातहरू बुझाउनुहोस्",
        "जग्गाको लालपुर्जा नामसारी गर्ने प्रक्रिया अनलाइन भएको छ",
        "राहदानीको लागि आवेदन दिने तरिका सरल बनाइएको छ",
        "मतदाता नामावलीमा नाम दर्ता गर्न निम्न कागजात चाहिन्छ",
    ],
    "Literary": [
        "नेपाली साहित्यमा कविता र गजलको विशेष स्थान छ",
        "पहाडको काखमा बसेको सानो गाउँमा एक किसानको परिवार थियो",
        "हिमालयको काखमा रहेको यो देश प्राकृतिक सम्पदाले भरिपूर्ण छ",
    ],
    "Technical": [
        "कम्प्युटर विज्ञानमा कृत्रिम बुद्धिमत्ताको प्रयोग बढ्दै छ",
        "सफ्टवेयर विकासमा नयाँ प्रविधिहरूको उपयोग हुन थालेको छ",
        "डिजिटल प्रविधिले नेपालको शिक्षा क्षेत्रमा क्रान्ति ल्याएको छ",
    ],
}

print("\n" + "="*70)
print("FINAL BENCHMARK: NepBPE-64k vs GPT-4o vs Llama3")
print("="*70)

totals = {"nepbpe": 0, "gpt4": 0, "llama3": 0}
sentence_count = 0
all_results = {}

for category, sentences in test_sentences.items():
    print(f"\n📂 {category}")
    print("-"*50)
    all_results[category] = []

    cat_totals = {"nepbpe": 0, "gpt4": 0, "llama3": 0}

    for sentence in sentences:
        nepbpe_count = len(sp.encode_as_pieces(sentence))
        gpt4_count   = len(gpt4.encode(sentence))
        llama_count  = len(llama3.encode(sentence))

        cat_totals["nepbpe"] += nepbpe_count
        cat_totals["gpt4"]   += gpt4_count
        cat_totals["llama3"] += llama_count

        totals["nepbpe"] += nepbpe_count
        totals["gpt4"]   += gpt4_count
        totals["llama3"] += llama_count
        sentence_count   += 1

        all_results[category].append({
            "sentence": sentence,
            "nepbpe": nepbpe_count,
            "gpt4": gpt4_count,
            "llama3": llama_count,
        })

    n = len(sentences)
    print(f"  NepBPE avg: {cat_totals['nepbpe']/n:.1f} tokens")
    print(f"  GPT-4o avg: {cat_totals['gpt4']/n:.1f} tokens")
    print(f"  Llama3 avg: {cat_totals['llama3']/n:.1f} tokens")
    print(f"  Saving vs GPT-4o: {((cat_totals['gpt4']-cat_totals['nepbpe'])/cat_totals['gpt4']*100):.1f}%")
    print(f"  Saving vs Llama3: {((cat_totals['llama3']-cat_totals['nepbpe'])/cat_totals['llama3']*100):.1f}%")

avg_nepbpe = totals["nepbpe"] / sentence_count
avg_gpt4   = totals["gpt4"]   / sentence_count
avg_llama3 = totals["llama3"] / sentence_count

saving_gpt4  = (avg_gpt4  - avg_nepbpe) / avg_gpt4  * 100
saving_llama = (avg_llama3 - avg_nepbpe) / avg_llama3 * 100

print("\n" + "="*70)
print("OVERALL RESULTS")
print("="*70)
print(f"\n  NepBPE-64k: {avg_nepbpe:.1f} tokens/sentence")
print(f"  GPT-4o:     {avg_gpt4:.1f} tokens/sentence")
print(f"  Llama3:     {avg_llama3:.1f} tokens/sentence")
print(f"\n  ✅ NepBPE uses {saving_gpt4:.1f}% fewer tokens than GPT-4o")
print(f"  ✅ NepBPE uses {saving_llama:.1f}% fewer tokens than Llama3")

print(f"\n  💰 Cost per 1M Nepali words (GPT-4o pricing $5/1M tokens):")
print(f"  GPT-4o:  ${5.00:.2f}")
print(f"  NepBPE:  ${5.00 * avg_nepbpe/avg_gpt4:.2f}")
print(f"  Savings: ${5.00 - 5.00 * avg_nepbpe/avg_gpt4:.2f} per million words")

summary = {
    "tokenizer": "NepBPE-SPM-64k",
    "vocab_size": 64000,
    "corpus": "Nepali Wikipedia (580k lines, 97MB raw)",
    "avg_tokens": {
        "nepbpe": avg_nepbpe,
        "gpt4o": avg_gpt4,
        "llama3": avg_llama3,
    },
    "savings_percent": {
        "vs_gpt4o": saving_gpt4,
        "vs_llama3": saving_llama,
    },
    "details": all_results
}

with open("benchmark/final_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n✅ Full results saved to benchmark/final_results.json")