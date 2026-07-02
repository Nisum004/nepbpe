import os
import json
import math
import numpy as np
import torch
import sentencepiece as spm
import tiktoken
from transformers import AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_torch import NepBPETorch, CONFIG

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="NepBPE API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load tokenizer ────────────────────────────────────────
print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/nepbpe_spm/nepbpe.model")
print(f"✅ NepBPE tokenizer loaded — vocab: {sp.vocab_size()}")

# ── Load GPT-4o tokenizer ─────────────────────────────────
gpt4_tok = tiktoken.get_encoding("cl100k_base")
print("✅ GPT-4o tokenizer loaded")

# ── Load Llama3 tokenizer ─────────────────────────────────
print("Loading Llama3 tokenizer...")
llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print("✅ Llama3 tokenizer loaded")

# ── Load models ───────────────────────────────────────────
device = "cpu"

def load_model(path):
    model = NepBPETorch(CONFIG)
    state = torch.load(path, map_location=device)
    # Map flat state dict to model
    model_state = model.state_dict()
    matched = {}
    for k in model_state:
        if k in state:
            matched[k] = state[k]
        else:
            print(f"  Missing key: {k}")
    model.load_state_dict(matched, strict=False)
    model.eval()
    return model

print("Loading main model...")
model = load_model("checkpoints/best_model.pt")
print("✅ Main model loaded")

print("Loading chat model...")
chat_model = load_model("checkpoints/chat_model.pt")
print("✅ Chat model loaded")

# ── Load benchmark ────────────────────────────────────────
with open("benchmark/final_results.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

print("🚀 NepBPE API ready!")

# ── Request models ────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    rep_penalty: float = 1.3

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8
    max_tokens: int = 100

# ── Generation helper ─────────────────────────────────────
def generate_tokens(mdl, input_ids, max_tokens, temperature, top_k, rep_penalty):
    generated = list(input_ids)
    x = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits      = mdl(x)
            next_logits = logits[0, -1, :] / temperature

            # Repetition penalty
            for prev_id in set(generated[-20:]):
                next_logits[prev_id] /= rep_penalty

            # Top-k
            top_vals, _ = torch.topk(next_logits, top_k)
            min_val      = top_vals[-1].item()
            next_logits[next_logits < min_val] = float("-inf")

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

            if next_token == sp.eos_id():
                break

    return generated

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "api": "NepBPE API", "version": "1.0"}

@app.post("/tokenize")
def tokenize(req: TextRequest):
    text = req.text.strip()
    if not text:
        return {"error": "Empty text"}

    nepbpe_pieces = sp.encode_as_pieces(text)
    nepbpe_ids    = sp.encode(text, out_type=int)
    gpt4_ids      = gpt4_tok.encode(text)
    gpt4_pieces   = [gpt4_tok.decode([t]) for t in gpt4_ids]
    llama_ids     = llama_tok.encode(text)
    llama_pieces  = llama_tok.convert_ids_to_tokens(llama_ids)

    def saving(base, ours):
        return round((base - ours) / base * 100, 1) if base > 0 else 0

    return {
        "text":   text,
        "nepbpe": {"tokens": nepbpe_pieces, "ids": nepbpe_ids, "count": len(nepbpe_pieces)},
        "gpt4o":  {"tokens": gpt4_pieces,   "ids": gpt4_ids,   "count": len(gpt4_ids)},
        "llama3": {"tokens": llama_pieces,   "ids": llama_ids,  "count": len(llama_ids)},
        "savings": {
            "vs_gpt4o":  saving(len(gpt4_ids),  len(nepbpe_pieces)),
            "vs_llama3": saving(len(llama_ids),  len(nepbpe_pieces)),
        },
        "cost": {
            "gpt4o_usd":  round(len(gpt4_ids)  / 1_000_000 * 5, 6),
            "nepbpe_usd": round(len(nepbpe_ids) / 1_000_000 * 5, 6),
        }
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    ids       = sp.encode(req.prompt, out_type=int)
    generated = generate_tokens(
        model, ids,
        req.max_tokens, req.temperature, req.top_k, req.rep_penalty
    )
    return {
        "prompt":    req.prompt,
        "generated": sp.decode(generated),
        "tokens":    len(generated),
    }

@app.post("/chat")
def chat(req: ChatRequest):
    full_prompt = f"प्रश्न । {req.message}\nउत्तर ।"
    ids         = sp.encode(full_prompt, out_type=int)
    stop_ids    = sp.encode("अन्त्य ।", out_type=int)
    new_q_ids   = sp.encode("प्रश्न ।", out_type=int)
    generated   = list(ids)
    x           = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(req.max_tokens):
            logits      = chat_model(x)
            next_logits = logits[0, -1, :] / req.temperature

            for prev_id in set(generated[-20:]):
                next_logits[prev_id] /= 1.4

            top_vals, _ = torch.topk(next_logits, 30)
            min_val      = top_vals[-1].item()
            next_logits[next_logits < min_val] = float("-inf")

            probs      = torch.softmax(next_logits, dim=-1)
            token_id   = torch.multinomial(probs, 1).item()
            generated.append(token_id)
            x = torch.cat([x, torch.tensor([[token_id]])], dim=1)

            if token_id == sp.eos_id():
                break
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids):] == stop_ids:
                generated = generated[:-len(stop_ids)]
                break
            if len(generated) >= len(new_q_ids) and generated[-len(new_q_ids):] == new_q_ids:
                generated = generated[:-len(new_q_ids)]
                break

    full_text = sp.decode(generated)
    answer    = full_text.split("उत्तर ।")[-1].strip() if "उत्तर ।" in full_text else full_text
    for stop in ["अन्त्य", "प्रश्न"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return {
        "message":  req.message,
        "response": answer if answer else "माफ गर्नुहोस्, मलाई थाहा छैन।",
        "tokens":   len(generated),
    }

@app.get("/benchmark")
def get_benchmark():
    return benchmark_data

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
