import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import sentencepiece as spm
import numpy as np
import time
import os
import json
import math

from model import NepBPE, CONFIG_117M

# ─── Config ───────────────────────────────────────────────
TRAIN_CONFIG = {
    "corpus":         "data/processed/nepali_clean.txt",
    "tokenizer":      "tokenizer/nepbpe_spm/nepbpe.model",
    "checkpoint_dir": "checkpoints",
    "batch_size":     16,     # down from 32
    "context_len":    512,    # down from 1024
    "lr":             3e-4,
    "min_lr":         1e-5,
    "warmup_steps":   200,
    "max_steps":      3000,   # down from 5000
    "eval_every":     100,
    "save_every":     500,
    "grad_clip":      1.0,
    "max_lines":      100000,
}

os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

# ─── Load tokenizer ───────────────────────────────────────
print("📖 Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(TRAIN_CONFIG["tokenizer"])
print(f"✅ Tokenizer loaded — vocab size: {sp.vocab_size()}")

# ─── Tokenize corpus ──────────────────────────────────────
print("📝 Tokenizing corpus...")
token_ids = []

with open(TRAIN_CONFIG["corpus"], "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    if line:
        ids = sp.encode(line, out_type=int)
        token_ids.extend(ids)
        token_ids.append(sp.eos_id())
    if i % 20000 == 0:
        print(f"  → {i:,} / {len(lines):,} lines tokenized "
              f"({len(token_ids):,} tokens so far)...")

token_ids = np.array(token_ids, dtype=np.int32)
print(f"✅ Total tokens: {len(token_ids):,}")

# ─── Train/val split ──────────────────────────────────────
split      = int(0.95 * len(token_ids))
train_data = token_ids[:split]
val_data   = token_ids[split:]
print(f"📊 Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

# ─── Batch sampler ────────────────────────────────────────
def get_batch(data, batch_size, context_len):
    ix = np.random.randint(0, len(data) - context_len - 1, size=batch_size)
    x  = np.stack([data[i:i + context_len]     for i in ix])
    y  = np.stack([data[i + 1:i + context_len + 1] for i in ix])
    return mx.array(x), mx.array(y)

# ─── Loss function ────────────────────────────────────────
def loss_fn(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    )
    return loss.mean()

# ─── LR scheduler (cosine with warmup) ───────────────────
def get_lr(step):
    if step < TRAIN_CONFIG["warmup_steps"]:
        return TRAIN_CONFIG["lr"] * step / TRAIN_CONFIG["warmup_steps"]
    progress = (step - TRAIN_CONFIG["warmup_steps"]) / \
               (TRAIN_CONFIG["max_steps"] - TRAIN_CONFIG["warmup_steps"])
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return TRAIN_CONFIG["min_lr"] + cosine * \
           (TRAIN_CONFIG["lr"] - TRAIN_CONFIG["min_lr"])

# ─── Build model ──────────────────────────────────────────
print("\n🏗️  Building NepBPE-184M model...")
model     = NepBPE(CONFIG_117M)
mx.eval(model.parameters())

total_params = sum(v.size for _, v in tree_flatten(model.parameters())
                   if isinstance(v, mx.array))
print(f"✅ Model ready — {total_params/1e6:.1f}M parameters")

# ─── Optimizer ────────────────────────────────────────────
optimizer = optim.AdamW(learning_rate=TRAIN_CONFIG["lr"], weight_decay=0.1)

loss_and_grad = nn.value_and_grad(model, loss_fn)

# ─── Eval ─────────────────────────────────────────────────
def evaluate(steps=20):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = get_batch(val_data,
                         TRAIN_CONFIG["batch_size"],
                         TRAIN_CONFIG["context_len"])
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

# ─── Generation sample ────────────────────────────────────
def generate_sample(prompt="नेपाल", max_tokens=50):
    model.eval()
    ids = sp.encode(prompt, out_type=int)
    x   = mx.array([ids])
    for _ in range(max_tokens):
        logits = model(x)
        next_logits = logits[:, -1, :]
        next_token  = mx.argmax(next_logits, axis=-1, keepdims=True)
        x = mx.concatenate([x, next_token], axis=1)
        if next_token.item() == sp.eos_id():
            break
    generated = x[0].tolist()
    model.train()
    return sp.decode(generated)

# ─── Training loop ────────────────────────────────────────
print("\n🚀 Starting training...")
print(f"   Batch size:    {TRAIN_CONFIG['batch_size']}")
print(f"   Context len:   {TRAIN_CONFIG['context_len']}")
print(f"   Max steps:     {TRAIN_CONFIG['max_steps']}")
print(f"   Eval every:    {TRAIN_CONFIG['eval_every']} steps")
print(f"   Save every:    {TRAIN_CONFIG['save_every']} steps")
print()

history  = []
t0       = time.time()
best_val = float("inf")

for step in range(TRAIN_CONFIG["max_steps"]):

    # Update LR
    lr = get_lr(step)
    optimizer.learning_rate = lr

    # Forward + backward
    x, y         = get_batch(train_data,
                              TRAIN_CONFIG["batch_size"],
                              TRAIN_CONFIG["context_len"])
    loss, grads  = loss_and_grad(model, x, y)

    # Gradient clipping
    grads, total_norm = optim.clip_grad_norm(grads, TRAIN_CONFIG["grad_clip"])

    # Update weights
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    # Clear MLX cache after every step — prevents memory accumulation
    mx.clear_cache()

    # ── Logging ──
    if step % 10 == 0:
        elapsed  = time.time() - t0
        tokens_s = (step + 1) * TRAIN_CONFIG["batch_size"] * \
                   TRAIN_CONFIG["context_len"] / elapsed
        print(f"Step {step:5d} | loss {loss.item():.4f} | "
              f"lr {lr:.2e} | {tokens_s:,.0f} tok/s")

    # ── Eval ──
    if step % TRAIN_CONFIG["eval_every"] == 0 and step > 0:
        val_loss = evaluate()
        print(f"\n{'─'*55}")
        print(f"  EVAL step {step} | val_loss {val_loss:.4f} | "
              f"perplexity {math.exp(val_loss):.2f}")
        sample = generate_sample("नेपाल")
        print(f"  Sample: {sample[:100]}")
        print(f"{'─'*55}\n")

        history.append({"step": step, "val_loss": val_loss,
                        "perplexity": math.exp(val_loss)})

        if val_loss < best_val:
            best_val = val_loss
            model.save_weights(
                f"{TRAIN_CONFIG['checkpoint_dir']}/best_model.npz")
            print(f"  💾 New best model saved (val_loss={best_val:.4f})")

    # ── Checkpoint ──
    if step % TRAIN_CONFIG["save_every"] == 0 and step > 0:
        model.save_weights(
            f"{TRAIN_CONFIG['checkpoint_dir']}/step_{step}.npz")
        print(f"💾 Checkpoint saved at step {step}")

# ─── Final save ───────────────────────────────────────────
model.save_weights(f"{TRAIN_CONFIG['checkpoint_dir']}/final_model.npz")
with open(f"{TRAIN_CONFIG['checkpoint_dir']}/history.json", "w") as f:
    json.dump(history, f, indent=2)

total_time = time.time() - t0
print(f"\n✅ Training complete in {total_time/3600:.2f} hours")
print(f"📊 Best val loss: {best_val:.4f} | Perplexity: {math.exp(best_val):.2f}")
print(f"💾 Model saved to checkpoints/final_model.npz")