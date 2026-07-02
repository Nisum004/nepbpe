import mlx.core as mx
import sentencepiece as spm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import NepBPE, CONFIG_117M

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")

model = NepBPE(CONFIG_117M)
model.load_weights("checkpoints/chat_model.npz")
mx.eval(model.parameters())
print("✅ Chat model loaded\n")

def chat(prompt, temperature=0.7, top_k=30, rep_penalty=1.4, max_tokens=100):
    # Use new format without : and ?
    full_prompt = f"प्रश्न । {prompt}\nउत्तर ।"
    ids = sp.encode(full_prompt, out_type=int)
    x   = mx.array([ids])
    generated = list(ids)

    # Stop token in new format
    stop_ids = sp.encode("अन्त्य ।", out_type=int)

    for _ in range(max_tokens):
        logits      = model(x)
        next_logits = logits[:, -1, :].squeeze(0) / temperature

        for prev_id in set(generated[-20:]):
            next_logits[prev_id] = next_logits[prev_id] / rep_penalty

        top_vals    = mx.topk(next_logits, top_k)
        min_val     = top_vals[-1].item()
        next_logits = mx.where(
            next_logits < min_val,
            mx.full(next_logits.shape, float("-inf")),
            next_logits
        )

        next_token  = mx.random.categorical(next_logits)
        token_id    = next_token.item()
        generated.append(token_id)
        x = mx.concatenate([x, next_token.reshape(1, 1)], axis=1)

        if token_id == sp.eos_id():
            break

        # Stop when model generates अन्त्य ।
        if len(generated) >= len(stop_ids):
            if generated[-len(stop_ids):] == stop_ids:
                generated = generated[:-len(stop_ids)]
                break

        # Also stop at new question
        new_q_ids = sp.encode("प्रश्न ।", out_type=int)
        if len(generated) >= len(new_q_ids):
            if generated[-len(new_q_ids):] == new_q_ids:
                generated = generated[:-len(new_q_ids)]
                break

    full_text = sp.decode(generated)

    # Extract answer
    if "उत्तर ।" in full_text:
        answer = full_text.split("उत्तर ।")[-1].strip()
        # Remove anything after अन्त्य
        if "अन्त्य" in answer:
            answer = answer.split("अन्त्य")[0].strip()
        if "प्रश्न" in answer:
            answer = answer.split("प्रश्न")[0].strip()
        return answer if answer else "माफ गर्नुहोस्, मलाई थाहा छैन।"
    return "माफ गर्नुहोस्, मलाई थाहा छैन।"

# Test questions
questions = [
    "नमस्ते, तपाईं कस्तो हुनुहुन्छ?",
    "नेपालको राजधानी के हो?",
    "सगरमाथाको उचाइ कति छ?",
    "नेपालको राष्ट्रिय फूल के हो?",
    "तपाईं को हुनुहुन्छ?",
    "दशैं के हो?",
    "कृत्रिम बुद्धिमत्ता के हो?",
    "नेपालमा कति जिल्ला छन्?",
    "पृथ्वीनारायण शाह को थिए?",
    "NepBPE के हो?",
]

print("="*60)
print("NepBPE Chat Model Test")
print("="*60)

for q in questions:
    print(f"\n🙋 {q}")
    print(f"🤖 {chat(q)}")

print("\n" + "="*60)
print("Interactive mode — type your question (or 'exit')")
print("="*60)

while True:
    q = input("\n🙋 तपाईंको प्रश्न: ").strip()
    if q.lower() in ["exit", "quit", "बाहिर"]:
        break
    if q:
        print(f"🤖 {chat(q)}")