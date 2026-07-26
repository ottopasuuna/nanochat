"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context), M=MSA (sparse, see below)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # MSA (MiniMax Sparse Attention, arXiv:2606.13392), used by layers marked 'M' above.
    # An MSA layer keeps a fixed attention budget of msa_top_k * msa_block_size tokens per
    # query, selected dynamically per GQA group by a lightweight index branch.
    msa_block_size: int = 64 # B_k: key block size for index selection
    msa_top_k: int = 8 # k: number of blocks selected per query per GQA group
    msa_idx_dim: int = 64 # d_idx: index head dimension
    msa_kl_weight: float = 1e-3 # lambda weight on the sum of per-layer index KL losses
    msa_warmup_frac: float = 0.01 # fraction of training steps with dense attention (indexer warmup)
    msa_idx_rope: bool = False # apply rotary embeddings to index q/k (ablation flag)


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, msa_dense=False, need_kl=False):
        # msa_dense/need_kl are accepted for API compatibility with MSAAttention and ignored here
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, None, None


class MSAAttention(nn.Module):
    """
    MiniMax Sparse Attention (arXiv:2606.13392). Two branches:

    - Index Branch: one index query head per GQA group + a single shared index key head
      (both computed from detached input). Scores the full causal context, max-pools
      scores to blocks of size B_k, and selects the top-k blocks per query per GQA group.
      The local block containing the query is always selected.
    - Main Branch: ordinary GQA softmax attention restricted to the selected blocks,
      with the selection shared by all query heads in a group.

    The index branch is trained with a KL loss aligning its distribution (over the
    selected tokens) to the detached, group-averaged Main Branch distribution.
    During indexer warmup (msa_dense=True) the Main Branch runs full attention and the
    KL is computed over the whole causal context; afterwards it runs block-sparse and
    the KL is restricted to the selected tokens.

    Reference implementation: plain PyTorch (torch.topk + masked SDPA). Correctness over
    speed; efficient inference and FlexAttention paths are future work (plan steps 5-6).
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.group_size = self.n_head // self.n_kv_head # G: query heads per GQA group
        self.block_size = config.msa_block_size
        self.top_k = config.msa_top_k
        self.idx_dim = config.msa_idx_dim
        self.idx_rope = config.msa_idx_rope
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert self.idx_dim % 2 == 0
        # Main branch projections (identical to CausalSelfAttention)
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Index branch projections: one index query head per GQA group, one shared index key head
        self.c_q_idx = Linear(self.n_embd, self.n_kv_head * self.idx_dim, bias=False)
        self.c_k_idx = Linear(self.n_embd, self.idx_dim, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, msa_dense=False, need_kl=False):
        B, T, C = x.size()

        # Main branch QKV projections (same recipe as CausalSelfAttention)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2
        k = k * 1.2

        if kv_cache is not None:
            # Inference fallback: dense attention over the full KV cache (correct, just not sparse).
            # TODO (plan step 5): true sparse decode with block selection + index-K cache.
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
            y = y.contiguous().view(B, T, -1)
            return self.c_proj(y), None, None

        # Index branch: detached input, so the KL loss trains only c_q_idx/c_k_idx
        x_det = x.detach()
        q_idx = self.c_q_idx(x_det).view(B, T, self.n_kv_head, self.idx_dim)
        k_idx = self.c_k_idx(x_det).view(B, T, self.idx_dim)
        if self.idx_rope:
            d = self.idx_dim // 2
            cos_idx, sin_idx = cos[..., :d], sin[..., :d]
            q_idx = apply_rotary_emb(q_idx, cos_idx, sin_idx)
            k_idx = apply_rotary_emb(k_idx.unsqueeze(2), cos_idx, sin_idx).squeeze(2)
        q_idx, k_idx = norm(q_idx), norm(k_idx)

        # Token-level index scores, causally masked: (B, H_kv, T_q, T_k)
        causal_bool = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        idx_scores = torch.einsum('bthd,bsd->bhts', q_idx, k_idx) * (self.idx_dim ** -0.5)
        idx_scores = idx_scores.masked_fill(~causal_bool, float('-inf'))

        if msa_dense:
            # Indexer warmup: full attention, KL over the whole causal context
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.contiguous().view(B, T, -1)
            sel_mask = causal_bool # (T, T), broadcast in the KL below
        else:
            # Block max-pool the index scores: (B, H_kv, T, n_blocks)
            Bk = self.block_size
            n_blocks = (T + Bk - 1) // Bk
            pad = n_blocks * Bk - T
            scores_padded = F.pad(idx_scores, (0, pad), value=float('-inf'))
            block_scores = scores_padded.view(B, self.n_kv_head, T, n_blocks, Bk).amax(dim=-1)
            # The local block containing the query is always selected (force its score to +inf)
            local_block = torch.arange(T, device=x.device) // Bk # (T,)
            block_scores.scatter_(-1, local_block.view(1, 1, T, 1).expand(B, self.n_kv_head, T, 1), float('inf'))
            # Top-k block selection per query per GQA group (exp-free: rank raw scores)
            k_sel = min(self.top_k, n_blocks)
            topk_idx = block_scores.topk(k_sel, dim=-1).indices # (B, H_kv, T, k_sel)
            sel_blocks = torch.zeros(B, self.n_kv_head, T, n_blocks, dtype=torch.bool, device=x.device)
            sel_blocks.scatter_(-1, topk_idx, True)
            # Expand block selection to token granularity and re-apply causality
            token_block = torch.arange(T, device=x.device) // Bk # (T_k,)
            sel_mask = sel_blocks[:, :, :, token_block] & causal_bool # (B, H_kv, T, T)
            # Main branch: SDPA over selected tokens only. Query heads are merged into the
            # GQA group dimension so the per-group mask broadcasts over the G heads in it.
            G = self.group_size
            q_g = q.view(B, T, self.n_kv_head, G, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B * self.n_kv_head, G, T, self.head_dim)
            k_g = k.permute(0, 2, 1, 3).unsqueeze(2).expand(B, self.n_kv_head, G, T, self.head_dim).reshape(B * self.n_kv_head, G, T, self.head_dim)
            v_g = v.permute(0, 2, 1, 3).unsqueeze(2).expand(B, self.n_kv_head, G, T, self.head_dim).reshape(B * self.n_kv_head, G, T, self.head_dim)
            mask_g = sel_mask.unsqueeze(2).reshape(B * self.n_kv_head, 1, T, T)
            y = F.scaled_dot_product_attention(q_g, k_g, v_g, attn_mask=mask_g) # default scale 1/sqrt(head_dim)
            y = y.view(B, self.n_kv_head, G, T, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, T, self.n_head * self.head_dim)

        # Auxiliary KL loss: align the index distribution with the detached, group-averaged
        # Main Branch distribution over the selected token set (all causal tokens in warmup)
        kl = None
        recall = None
        if need_kl:
            G = self.group_size
            with torch.no_grad():
                # Main branch logits over the full causal context: (B, H_kv, G, T, T)
                main_logits = torch.einsum('bthgd,bshd->bhgts', q.view(B, T, self.n_kv_head, G, self.head_dim), k) * (self.head_dim ** -0.5)
                main_logits = main_logits.masked_fill(~causal_bool, float('-inf'))
                if not msa_dense:
                    # Block recall diagnostic (from the paper): overlap between the index
                    # branch's top-k blocks and the top-k blocks induced by the main branch
                    main_blk = main_logits.mean(dim=2) # group-averaged scores: (B, H_kv, T, T)
                    main_blk = F.pad(main_blk, (0, pad), value=float('-inf')).view(B, self.n_kv_head, T, n_blocks, Bk).amax(dim=-1)
                    main_blk.scatter_(-1, local_block.view(1, 1, T, 1).expand(B, self.n_kv_head, T, 1), float('inf'))
                    main_topk = main_blk.topk(k_sel, dim=-1).indices # (B, H_kv, T, k_sel)
                    inter = (main_topk.unsqueeze(-1) == topk_idx.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
                    recall = (inter.float() / k_sel).mean()
                # Restrict to the selected token set for the teacher distribution
                teacher_mask = sel_mask.unsqueeze(2) if sel_mask.dim() == 4 else sel_mask
                main_logits = main_logits.masked_fill(~teacher_mask, float('-inf'))
                teacher_p = F.log_softmax(main_logits.float(), dim=-1).exp().mean(dim=2) # average over G heads
            student_logp = F.log_softmax(idx_scores.masked_fill(~sel_mask, float('-inf')).float(), dim=-1)
            # Zero out log-probs outside the selected set: teacher_p is 0 there, and
            # kl_div's target*input term would otherwise produce 0*(-inf) = NaN
            student_logp = student_logp.masked_fill(~sel_mask, 0.0)
            kl = F.kl_div(student_logp, teacher_p, reduction='none').sum(dim=-1).mean()

        y = self.c_proj(y)
        return y, kl, recall


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_msa=False):
        super().__init__()
        self.attn = MSAAttention(config, layer_idx) if use_msa else CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, msa_dense=False, need_kl=False):
        attn_out, kl, recall = self.attn(norm(x), ve, cos_sin, window_size, kv_cache, msa_dense=msa_dense, need_kl=need_kl)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, kl, recall


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes, self.msa_layers = self._compute_window_sizes(config)
        self.has_msa = any(self.msa_layers)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, use_msa=self.msa_layers[layer_idx]) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.attn, MSAAttention):
                torch.nn.init.uniform_(block.attn.c_q_idx.weight, -s, s) # index branch: init like c_q/c_k
                torch.nn.init.uniform_(block.attn.c_k_idx.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns (window_sizes, msa_layers):
        - window_sizes: list of (left, right) tuples for FA3's window_size parameter:
          left = how many tokens before current position to attend to (-1 = unlimited),
          right = how many tokens after current position to attend to (0 for causal)
        - msa_layers: list of bools, True for layers using MSA sparse attention

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context), M=MSA (sparse;
        gets a full-context window since it does its own block selection internally)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SLM" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S, L and M."
        if "M" in pattern:
            assert config.msa_block_size * config.msa_top_k <= config.sequence_len, (
                f"MSA budget msa_top_k*msa_block_size ({config.msa_top_k * config.msa_block_size}) "
                f"exceeds sequence_len ({config.sequence_len}): sparsity would be a no-op")
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
            "M": (long_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        msa_layers = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if layer_idx == config.n_layer - 1:
                char = "L" # final layer always gets full-context dense attention
            window_sizes.append(char_to_window[char])
            msa_layers.append(char == "M")
        return window_sizes, msa_layers

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window / MSA sparsity
        attn_flops = 0
        for window_size, is_msa in zip(self.window_sizes, self.msa_layers):
            if is_msa:
                attn_flops += self._msa_attn_flops_per_token()
            else:
                window = window_size[0]  # (left, right) tuple, we use left
                effective_seq = t if window < 0 else min(window, t)
                attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def _msa_budget(self, t=None):
        """Number of tokens each query attends to in an MSA layer (the fixed selection budget)."""
        t = t or self.config.sequence_len
        return min(self.config.msa_top_k * self.config.msa_block_size, t)

    def _msa_attn_flops_per_token(self):
        """
        Algorithmic FLOPs per token (fwd+bwd) for one MSA layer, following the paper's
        complexity model (arXiv:2606.13392): index branch scores over the full causal
        context (QK^T only, 2*H_kv*d_idx*t forward) + main branch over the k*B_k selected
        tokens (4*H_q*d_h*budget forward), times 3 for fwd+bwd, plus the forward-only
        teacher logits pass (2*H_q*d_h*t) used by the KL loss during training.
        Note: during indexer warmup MSA layers are dense (12*h*q*t), and the current
        masked-SDPA reference path executes dense attention; this reports the algorithmic
        cost that the sparse implementation (FlexAttention, plan step 6) realizes.
        """
        c = self.config
        h, q, t = c.n_head, c.n_embd // c.n_head, c.sequence_len
        budget = self._msa_budget(t)
        idx_flops = 2 * c.n_kv_head * c.msa_idx_dim * t
        main_flops = 4 * h * q * budget
        teacher_flops = 2 * h * q * t # KL teacher logits, forward only (no_grad)
        return 3 * (idx_flops + main_flops) + teacher_flops

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for (window, _), is_msa in zip(self.window_sizes, self.msa_layers):
            if is_msa:
                # sparse decode (plan step 5): index pass over full context + selected blocks
                attn_flops += 4 * h * q * min(context_len, self._msa_budget()) \
                              + 2 * self.config.n_kv_head * self.config.msa_idx_dim * context_len
            else:
                attn_flops += 4 * h * q * min(context_len, window)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for (window, _), is_msa in zip(self.window_sizes, self.msa_layers):
            if is_msa:
                # index branch scores: causal, so 2*H_kv*d_idx per (query, key) pair
                attn_flops += 2 * self.config.n_kv_head * self.config.msa_idx_dim * num_tokens * (num_tokens + 1) // 2
                # main branch: attends to min(budget, t) tokens at position t
                w = min(self._msa_budget(), num_tokens)
                attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w
                attn_flops += 4 * h * q * attended_tokens
            else:
                w = min(window, num_tokens)
                attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
                attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for (window, _), is_msa in zip(self.window_sizes, self.msa_layers):
            if is_msa:
                # sparse decode (plan step 5): read selected blocks + the index-K cache
                total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, self._msa_budget())
                total += self.config.msa_idx_dim * kv_dtype_bytes * context_len # single shared index key head
            else:
                total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', msa_dense=False, return_kl=False):
        # msa_dense: run MSA layers with full attention (indexer warmup phase of training)
        # return_kl: also compute and return the sum of per-layer MSA index KL losses
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        kl_total = None # sum of per-layer MSA index KL losses (if any MSA layers)
        recall_total = None # mean of per-layer MSA block recall diagnostics
        n_recall = 0
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x, kl, recall = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, msa_dense=msa_dense, need_kl=return_kl)
            if kl is not None:
                kl_total = kl if kl_total is None else kl_total + kl
            if recall is not None:
                recall_total = recall if recall_total is None else recall_total + recall
                n_recall += 1
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_kl:
                if kl_total is None:
                    kl_total = logits.new_zeros(())
                if recall_total is None:
                    recall_total = logits.new_zeros(())
                else:
                    recall_total = recall_total / n_recall
                return loss, kl_total, recall_total
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
