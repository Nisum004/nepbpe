import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import sentencepiece as spm
import numpy as np
import time
import os
import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NepBPE, CONFIG_117M

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FINETUNE_CONFIG = {
    "chat_data":      "data/chat_data.txt",
    "tokenizer":      "tokenizer/nepbpe_spm/nepbpe.model",
    "base_model":     "checkpoints/best_model.npz",
    "save_path":      "checkpoints/chat_model.npz",
    "batch_size":     4,
    "context_len":    256,
    "lr":             5e-5,    # lower LR for fine-tuning
    "max_steps":      1000,
    "eval_every":     50,
    "grad_clip":      1.0,
}

# Load tokenizer
print("📖 Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(FINETUNE_CONFIG["tokenizer"])

# Load chat data
print("📝 Tokenizing chat data...")
token_ids = []
with open(FINETUNE_CONFIG["chat_data"], "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize full text
ids = sp.encode(text, out_type=int)
token_ids = np.array(ids, dtype=np.int32)
print(f"✅ {len(token_ids):,} tokens from chat data")

# Train/val split
split      = int(0.9 * len(token_ids))
train_data = token_ids[:split]
val_data   = token_ids[split:]

def get_batch(data, batch_size, context_len):
    if len(data) <= context_len:
        # Repeat data if too small
        data = np.tile(data, 10)
    ix = np.random.randint(0, len(data) - context_len - 1, size=batch_size)
    x  = np.stack([data[i:i + context_len]         for i in ix])
    y  = np.stack([data[i + 1:i + context_len + 1] for i in ix])
    return mx.array(x), mx.array(y)

def loss_fn(model, x, y):
    logits  = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    ).mean()

# Load pretrained model
print("🏗️  Loading pretrained model...")
model = NepBPE(CONFIG_117M)
model.load_weights(FINETUNE_CONFIG["base_model"])
mx.eval(model.parameters())
print("✅ Pretrained weights loaded")

optimizer     = optim.AdamW(
    learning_rate=FINETUNE_CONFIG["lr"],
    weight_decay=0.01
)
loss_and_grad = nn.value_and_grad(model, loss_fn)

def chat_generate(prompt, max_tokens=80, temperature=0.7, top_k=30, rep_penalty=1.3):
    full_prompt = f"प्रश्न । {prompt}\nउत्तर ।"
    ids = sp.encode(full_prompt, out_type=int)
    x   = mx.array([ids])
    generated = list(ids)

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

        next_token = mx.random.categorical(next_logits)
        token_id   = next_token.item()
        generated.append(token_id)
        x = mx.concatenate([x, next_token.reshape(1, 1)], axis=1)

        if token_id == sp.eos_id():
            break

    full_text = sp.decode(generated)
    # Extract just the answer part
    if "उत्तर ।" in full_text:
        answer = full_text.split("उत्तर ।")[-1].strip()
        return answer
    return full_text

print(f"\n🚀 Fine-tuning on chat data...")
print(f"   Steps: {FINETUNE_CONFIG['max_steps']}")
print(f"   LR:    {FINETUNE_CONFIG['lr']}\n")

best_val = float("inf")
t0       = time.time()

for step in range(FINETUNE_CONFIG["max_steps"]):

    x, y        = get_batch(
        train_data,
        FINETUNE_CONFIG["batch_size"],
        FINETUNE_CONFIG["context_len"]
    )
    loss, grads = loss_and_grad(model, x, y)
    grads, _    = optim.clip_grad_norm(grads, FINETUNE_CONFIG["grad_clip"])
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    mx.clear_cache()

    if step % 10 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

    if step % FINETUNE_CONFIG["eval_every"] == 0 and step > 0:
        # Test chat
        test_questions = [
            "नेपालको राजधानी के हो?",
            "तपाईं को हुनुहुन्छ?",
            "सगरमाथाको उचाइ कति छ?",
        ]
        print(f"\n{'─'*50}")
        print(f"CHAT TEST — step {step}")
        for q in test_questions:
            ans = chat_generate(q)
            print(f"Q: {q}")
            print(f"A: {ans[:100]}")
            print()
        print(f"{'─'*50}\n")

        # Save if improving
        val_loss_val = loss.item()
        if val_loss_val < best_val:
            best_val = val_loss_val
            model.save_weights(FINETUNE_CONFIG["save_path"])
            print(f"💾 Chat model saved")

# Final save
model.save_weights(FINETUNE_CONFIG["save_path"])
print(f"\n✅ Fine-tuning complete in {(time.time()-t0)/60:.1f} minutes")
print(f"💾 Chat model: {FINETUNE_CONFIG['save_path']}")