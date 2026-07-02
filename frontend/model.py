import mlx.core as mx
import mlx.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["d_model"] % config["n_heads"] == 0

        self.n_heads = config["n_heads"]
        self.d_model = config["d_model"]
        self.head_dim = config["d_model"] // config["n_heads"]

        self.qkv_proj = nn.Linear(config["d_model"], 3 * config["d_model"], bias=False)
        self.out_proj  = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.dropout   = nn.Dropout(config["dropout"])

    def __call__(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Causal mask
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = nn.Linear(config["d_model"], 4 * config["d_model"], bias=False)
        self.fc2     = nn.Linear(4 * config["d_model"], config["d_model"], bias=False)
        self.dropout = nn.Dropout(config["dropout"])
        self.gelu    = nn.GELU()

    def __call__(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config["d_model"])
        self.ln2  = nn.LayerNorm(config["d_model"])
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NepBPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb    = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_emb = nn.Embedding(config["context_len"], config["d_model"])
        self.dropout      = nn.Dropout(config["dropout"])
        self.blocks       = [TransformerBlock(config) for _ in range(config["n_layers"])]
        self.ln_final     = nn.LayerNorm(config["d_model"])
        self.lm_head      = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

    def __call__(self, x):
        B, T = x.shape
        positions = mx.arange(T)

        tok = self.token_emb(x)
        pos = self.position_emb(positions)
        h   = self.dropout(tok + pos)

        for block in self.blocks:
            h = block(h)

        h      = self.ln_final(h)
        logits = self.lm_head(h)
        return logits

    def count_params(self):
        total = sum(p.size for p in self.parameters().values()
                    if isinstance(p, mx.array))
        return total


# 117M config — full send
CONFIG_117M = {
    "vocab_size":   64000,
    "context_len":  1024,
    "d_model":      768,
    "n_layers":     12,
    "n_heads":      12,
    "dropout":      0.1,
}
    
if __name__ == "__main__":
    from mlx.utils import tree_flatten

    print("🏗️  Building NepBPE-117M...")
    model = NepBPE(CONFIG_117M)

    # Count parameters
    total = sum(
        v.size for _, v in tree_flatten(model.parameters())
        if isinstance(v, mx.array)
    )

    # Test forward pass
    dummy = mx.zeros((2, 16), dtype=mx.int32)
    out   = model(dummy)
    mx.eval(out)

    print(f"✅ Model built successfully")
    print(f"📐 Output shape: {out.shape}")
    print(f"🔢 Parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"📋 Config: {CONFIG_117M}")