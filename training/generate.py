import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from model import NepBPE, CONFIG_117M

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")

# Load best model
model = NepBPE(CONFIG_117M)
model.load_weights("checkpoints/best_model.npz")
mx.eval(model.parameters())
print("✅ Model loaded\n")

def generate(prompt, max_tokens=100, temperature=0.9, top_k=50, rep_penalty=1.3):
    ids = sp.encode(prompt, out_type=int)
    x   = mx.array([ids])
    generated_ids = list(ids)

    for _ in range(max_tokens):
        logits      = model(x)
        next_logits = logits[:, -1, :].squeeze(0) / temperature

        # Repetition penalty — penalize already seen tokens
        for prev_id in set(generated_ids[-20:]):
            next_logits[prev_id] = next_logits[prev_id] / rep_penalty

        # Top-k sampling
        top_vals  = mx.topk(next_logits, top_k)
        min_val   = top_vals[-1].item()
        next_logits = mx.where(
            next_logits < min_val,
            mx.full(next_logits.shape, float("-inf")),
            next_logits
        )

        next_token = mx.random.categorical(next_logits)
        token_id   = next_token.item()
        generated_ids.append(token_id)

        next_token = next_token.reshape(1, 1)
        x          = mx.concatenate([x, next_token], axis=1)

        if token_id == sp.eos_id():
            break

    return sp.decode(generated_ids)

# Test prompts
prompts = [
    "नेपाल",
    "काठमाडौं",
    "नेपाली भाषा",
    "नेपाल सरकारले",
    "हिमालयको",
]

print("="*60)
print("NepBPE-184M Generation Samples")
print("="*60)

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {generate(prompt)}")
    print("-"*40)