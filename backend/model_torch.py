import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads  = config["n_heads"]
        self.d_model  = config["d_model"]
        self.head_dim = config["d_model"] // config["n_heads"]
        self.qkv_proj = nn.Linear(config["d_model"], 3 * config["d_model"], bias=False)
        self.out_proj  = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.dropout   = nn.Dropout(config["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale
        mask   = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        attn   = torch.softmax(scores, dim=-1)
        out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = nn.Linear(config["d_model"], 4 * config["d_model"], bias=False)
        self.fc2     = nn.Linear(4 * config["d_model"], config["d_model"], bias=False)
        self.dropout = nn.Dropout(config["dropout"])
        self.gelu    = nn.GELU()

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config["d_model"])
        self.ln2  = nn.LayerNorm(config["d_model"])
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NepBPETorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb    = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_emb = nn.Embedding(config["context_len"], config["d_model"])
        self.dropout      = nn.Dropout(config["dropout"])
        self.blocks       = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])])
        self.ln_final     = nn.LayerNorm(config["d_model"])
        self.lm_head      = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        h    = self.dropout(self.token_emb(x) + self.position_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.ln_final(h))

CONFIG = {
    "vocab_size":  64000,
    "context_len": 1024,
    "d_model":     768,
    "n_layers":    12,
    "n_heads":     12,
    "dropout":     0.0,
}
