from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import look_around
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Mlp_prev(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd*(config.n_prev+1), config.mlp_ratio * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_prev = config.n_prev

    def forward(self, x):
        x = look_around(x, backward = self.n_prev, forward = 0, pad_value = 0, dim = 2)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        if config.n_prev > 0:
            self.mlp = Mlp_prev(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    mlp_ratio: int = 4 # mlp hidden dimension = mlp_ratio * n_embd
    n_prev: int = 1 # number of previous tokens to consider

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_dict(self):
        return self.__dict__

class GPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = GPTConfig.from_dict(config)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd),
            wpe = nn.Embedding(self.cfg.block_size, self.cfg.n_embd),
            h = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)]),
            ln_f = nn.LayerNorm(self.cfg.n_embd),
        ))
        self.lm_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.cfg.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"Cannot forward sequence of length {T}, block size is only {self.cfg.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

if __name__ == "__main__":
    # Create a small config for testing
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        mlp_ratio=4,
        n_prev=1
    )

    # Instantiate the model
    model = GPT(config.to_dict())
    print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters")

    # Create a sample input
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Forward pass
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")

    # Test with targets
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    logits = model(input_ids, targets)

    print("Basic checks passed!")