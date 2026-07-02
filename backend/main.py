from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sentencepiece as spm
import tiktoken
from transformers import AutoTokenizer
import mlx.core as mx
import sys
import os
import json

# Fix paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import NepBPE, CONFIG_117M

app = FastAPI(title="NepBPE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load all tokenizers ────────────────────────────────────
print("📖 Loading tokenizers...")

# NepBPE
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")
print("✅ NepBPE tokenizer loaded")

# GPT-4o
gpt4_tok = tiktoken.get_encoding("cl100k_base")
print("✅ GPT-4o tokenizer loaded")

# Llama3
print("Loading Llama3 tokenizer...")
llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print("✅ Llama3 tokenizer loaded")

# ── Load model ────────────────────────────────────────────
print("🏗️  Loading NepBPE-184M model...")
model = NepBPE(CONFIG_117M)
model.load_weights("checkpoints/best_model.npz")
mx.eval(model.parameters())
print("✅ Model loaded")
# Add after existing model loading
print("🤖 Loading chat model...")
chat_model = NepBPE(CONFIG_117M)
chat_model.load_weights("checkpoints/chat_model.npz")
mx.eval(chat_model.parameters())
print("✅ Chat model loaded")
# ── Load benchmark results ────────────────────────────────
with open("benchmark/final_results.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

print("\n🚀 NepBPE API ready!\n")

# ── Request models ────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    rep_penalty: float = 1.3

# ── Tokenize endpoint ─────────────────────────────────────
@app.post("/tokenize")
def tokenize(req: TextRequest):
    text = req.text.strip()
    if not text:
        return {"error": "Empty text"}

    # NepBPE tokens
    nepbpe_pieces = sp.encode_as_pieces(text)
    nepbpe_ids    = sp.encode(text, out_type=int)

    # GPT-4o tokens
    gpt4_ids    = gpt4_tok.encode(text)
    gpt4_pieces = [gpt4_tok.decode([t]) for t in gpt4_ids]

    # Llama3 tokens
    llama_ids    = llama_tok.encode(text)
    llama_pieces = llama_tok.convert_ids_to_tokens(llama_ids)

    # Savings
    def saving(base, ours):
        return round((base - ours) / base * 100, 1) if base > 0 else 0

    return {
        "text": text,
        "nepbpe": {
            "tokens": nepbpe_pieces,
            "ids":    nepbpe_ids,
            "count":  len(nepbpe_pieces),
        },
        "gpt4o": {
            "tokens": gpt4_pieces,
            "ids":    gpt4_ids,
            "count":  len(gpt4_ids),
        },
        "llama3": {
            "tokens": llama_pieces,
            "ids":    llama_ids,
            "count":  len(llama_ids),
        },
        "savings": {
            "vs_gpt4o":  saving(len(gpt4_ids),   len(nepbpe_pieces)),
            "vs_llama3": saving(len(llama_ids),   len(nepbpe_pieces)),
        },
        "cost": {
            "gpt4o_usd":  round(len(gpt4_ids)   / 1_000_000 * 5, 6),
            "nepbpe_usd": round(len(nepbpe_ids)  / 1_000_000 * 5, 6),
        }
    }

# ── Generate endpoint ─────────────────────────────────────
@app.post("/generate")
def generate(req: GenerateRequest):
    ids = sp.encode(req.prompt, out_type=int)
    x   = mx.array([ids])
    generated_ids = list(ids)

    for _ in range(req.max_tokens):
        logits      = model(x)
        next_logits = logits[:, -1, :].squeeze(0) / req.temperature

        # Repetition penalty
        for prev_id in set(generated_ids[-20:]):
            next_logits[prev_id] = next_logits[prev_id] / req.rep_penalty

        # Top-k
        top_vals    = mx.topk(next_logits, req.top_k)
        min_val     = top_vals[-1].item()
        next_logits = mx.where(
            next_logits < min_val,
            mx.full(next_logits.shape, float("-inf")),
            next_logits
        )

        next_token = mx.random.categorical(next_logits)
        token_id   = next_token.item()
        generated_ids.append(token_id)

        x = mx.concatenate([x, next_token.reshape(1, 1)], axis=1)

        if token_id == sp.eos_id():
            break

    return {
        "prompt":    req.prompt,
        "generated": sp.decode(generated_ids),
        "tokens":    len(generated_ids),
    }

# ── Benchmark endpoint ────────────────────────────────────
@app.get("/benchmark")
def get_benchmark():
    return benchmark_data

# ── Stats endpoint ────────────────────────────────────────
@app.get("/stats")
def get_stats():
    return {
        "model":        "NepBPE-184M",
        "parameters":   "184M",
        "vocab_size":   64000,
        "architecture": "GPT-2 style Transformer",
        "tokenizer":    "SentencePiece BPE",
        "corpus":       "Nepali Wikipedia (580k lines)",
        "savings": {
            "vs_gpt4o":  "83.9%",
            "vs_llama3": "71.5%",
        },
        "huggingface": {
            "tokenizer": "https://huggingface.co/nisum04/nepbpe-tokenizer",
            "model":     "https://huggingface.co/nisum04/nepbpe-184m",
        }
    }

# ── Health check ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "api": "NepBPE API", "version": "1.0"}

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8
    max_tokens: int = 100

# -- Chat endpoint ─────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    full_prompt = f"प्रश्न । {req.message}\nउत्तर ।"
    ids = sp.encode(full_prompt, out_type=int)
    x   = mx.array([ids])
    generated = list(ids)
    rep_penalty = 1.4
    top_k = 30

    stop_ids  = sp.encode("अन्त्य ।", out_type=int)
    new_q_ids = sp.encode("प्रश्न ।", out_type=int)

    for _ in range(req.max_tokens):
        logits      = chat_model(x)
        next_logits = logits[:, -1, :].squeeze(0) / req.temperature

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

        if len(generated) >= len(stop_ids):
            if generated[-len(stop_ids):] == stop_ids:
                generated = generated[:-len(stop_ids)]
                break

        if len(generated) >= len(new_q_ids):
            if generated[-len(new_q_ids):] == new_q_ids:
                generated = generated[:-len(new_q_ids)]
                break

    full_text = sp.decode(generated)
    answer = full_text.split("उत्तर ।")[-1].strip() if "उत्तर ।" in full_text else full_text

    for stop in ["अन्त्य", "प्रश्न"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return {
        "message":  req.message,
        "response": answer if answer else "माफ गर्नुहोस्, मलाई थाहा छैन।",
        "tokens":   len(generated),
    }