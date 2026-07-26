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
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # NSA (Native Sparse Attention) configuration. Set nsa_block_size > 0 to enable.
    nsa_block_size: int = 0   # compression block size (0 = disable NSA, use standard attention)
    nsa_stride: int = 0       # stride between compression blocks (0 = defaults to nsa_block_size)
    nsa_top_k_blocks: int = 4 # number of blocks to select for fine-grained attention
    nsa_window_size: int = 256 # sliding window size for local branch
    nsa_cmp_mlp_ratio: int = 4 # compression MLP hidden dim = n_embd * ratio


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

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
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
        return y


class NSACausalSelfAttention(nn.Module):
    """Native Sparse Attention (NSA) — three-branch hierarchical sparse attention.

    Branches:
      1. Compression: learned block-level MLP compression + causal attention over compressed tokens.
         Produces per-position block importance scores as a free byproduct.
      2. Selection: top-k blocks (chosen by compression scores) attended with fine-grained tokens.
      3. Sliding window: standard causal attention over the most recent window_size tokens.

    The three branch outputs are combined via a learned per-position gate.

    Ref: https://arxiv.org/abs/2502.11089
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.block_size = config.nsa_block_size
        self.stride = config.nsa_stride if config.nsa_stride > 0 else config.nsa_block_size
        self.top_k_blocks = config.nsa_top_k_blocks
        self.window_size = config.nsa_window_size

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Compression branch: learned block MLP + own Q/K/V projections
        cmp_hidden = self.n_embd * config.nsa_cmp_mlp_ratio
        self.cmp_mlp = nn.Sequential(
            nn.LayerNorm(self.block_size * self.n_embd),
            Linear(self.block_size * self.n_embd, cmp_hidden),
            nn.ReLU(),
            Linear(cmp_hidden, self.n_embd),
        )
        self.cmp_pos = nn.Embedding(self.block_size, self.n_embd)
        self.cmp_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.cmp_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.cmp_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Selection branch: own Q/K/V projections (shares selection with compression scores)
        self.slc_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.slc_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.slc_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Window branch: own Q/K/V projections (independent KV to prevent gradient interference)
        self.win_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.win_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.win_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Gated combination of three branches
        self.gate_mlp = nn.Sequential(
            Linear(self.n_embd, self.n_embd // 4),
            nn.ReLU(),
            Linear(self.n_embd // 4, 3),
        )

        # Shared output projection
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        # Value embedding gate (same interface as CausalSelfAttention)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size_unused, kv_cache):
        B, T, C = x.size()
        assert kv_cache is None, "NSA does not yet support KV cache inference"

        # Compute Q, K, V for each branch (independent projections)
        cmp_q = self.cmp_q(x).view(B, T, self.n_head, self.head_dim)
        slc_q = self.slc_q(x).view(B, T, self.n_head, self.head_dim)
        win_q = self.win_q(x).view(B, T, self.n_head, self.head_dim)
        k_raw = self.win_k(x).view(B, T, self.n_kv_head, self.head_dim)  # for compression input
        slc_k = self.slc_k(x).view(B, T, self.n_kv_head, self.head_dim)
        slc_v = self.slc_v(x).view(B, T, self.n_kv_head, self.head_dim)
        win_k = self.win_k(x).view(B, T, self.n_kv_head, self.head_dim)
        win_v = self.win_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply value embedding to window branch values
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            win_v = win_v + gate.unsqueeze(-1) * ve

        # Apply RoPE and QK norm to all queries and keys
        cos, sin = cos_sin
        cmp_q = norm(apply_rotary_emb(cmp_q, cos, sin)) * 1.2
        slc_q = norm(apply_rotary_emb(slc_q, cos, sin)) * 1.2
        win_q = norm(apply_rotary_emb(win_q, cos, sin)) * 1.2
        slc_k = norm(apply_rotary_emb(slc_k, cos, sin)) * 1.2
        win_k = norm(apply_rotary_emb(win_k, cos, sin)) * 1.2

        # ================================================================
        # Branch 1: Compression (coarse global context + block importance scores)
        # ================================================================
        num_cmp = max(0, (T - self.block_size) // self.stride + 1)

        if num_cmp > 0:
            # Build compression blocks: (B, num_cmp, block_size * C)
            positions = torch.arange(0, num_cmp * self.stride, self.stride, device=x.device)
            offsets = torch.arange(self.block_size, device=x.device)
            idx_cmp = (positions.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(max=T - 1)
            blocks_cmp = x[:, idx_cmp]  # (B, num_cmp, block_size, C)
            blocks_flat = blocks_cmp.reshape(B, num_cmp, self.block_size * self.n_embd)

            # Add position encoding and compress via MLP
            pos_enc = self.cmp_pos(offsets).view(1, 1, self.block_size * self.n_embd)
            blocks_flat = blocks_flat + pos_enc

            # Mask out padding positions (for partial last block)
            valid_len = T - positions  # (num_cmp,)
            valid_len = valid_len.clamp(max=self.block_size)
            pad_mask_cmp = (offsets.unsqueeze(0) < valid_len.unsqueeze(-1)).to(x.dtype)  # (num_cmp, block_size)
            pad_mask_cmp = pad_mask_cmp.reshape(1, num_cmp, self.block_size, 1)  # broadcast over C
            blocks_flat = blocks_flat.reshape(B, num_cmp, self.block_size, self.n_embd) * pad_mask_cmp
            blocks_flat = blocks_flat.reshape(B, num_cmp, self.block_size * self.n_embd)

            # Compress to one vector per block, cast back to model dtype
            cmp_out = self.cmp_mlp(blocks_flat).to(x.dtype)  # (B, num_cmp, C)
            cmp_k = self.cmp_k(cmp_out).view(B, num_cmp, self.n_kv_head, self.head_dim)
            cmp_v = self.cmp_v(cmp_out).view(B, num_cmp, self.n_kv_head, self.head_dim)

            # Compute attention scores (manual, to extract per-position scores for selection)
            # Compute per-KV-head scores, then repeat for each query head in the group
            # cmp_q: (B, T, H, D), cmp_k: (B, num_cmp, H_kv, D)
            heads_per_group = self.n_head // self.n_kv_head
            # Average query heads within each group: (B, T, H_kv, D)
            cmp_q_grouped = cmp_q.reshape(B, T, self.n_kv_head, heads_per_group, self.head_dim).mean(dim=3)
            # Scores: (B, H_kv, T, num_cmp)
            cmp_scores_grouped = torch.einsum(
                'bthd,bshd->bhts',
                cmp_q_grouped,
                cmp_k,
            )
            # Repeat scores for all query heads in each group: (B, H, T, num_cmp)
            cmp_scores = cmp_scores_grouped.repeat_interleave(heads_per_group, dim=1)

            # Causal mask: query at position t can only attend to compression tokens with
            # block end <= t. Block i ends at position (i * stride + block_size - 1).
            block_end = positions + self.block_size - 1  # (num_cmp,)
            causal_ok = block_end.unsqueeze(0) < T  # (T, num_cmp) via broadcasting
            causal_ok = causal_ok.unsqueeze(0).unsqueeze(0)  # (1, 1, T, num_cmp)

            # Padding mask: exclude compression tokens from partial blocks
            has_content = (torch.arange(num_cmp, device=x.device) < num_cmp).float()
            cmp_pad_ok = has_content.view(1, 1, 1, num_cmp)

            combined_mask = causal_ok.float() * cmp_pad_ok
            cmp_scores = cmp_scores.masked_fill(combined_mask == 0, float('-inf'))

            # Compute per-position importance scores (before softmax, for block selection)
            # Sum across query heads within each GQA group for shared block selection
            cmp_importance = cmp_scores.detach().sum(dim=1)  # (B, T, num_cmp)

            # Softmax and compute compression branch output (GQA-aware)
            cmp_attn = F.softmax(cmp_scores, dim=-1)  # (B, H, T, num_cmp)
            # Average attention weights within each group: (B, H_kv, T, num_cmp)
            cmp_attn_grouped = cmp_attn.reshape(B, self.n_kv_head, heads_per_group, T, num_cmp).mean(dim=2)
            # cmp_v: (B, num_cmp, H_kv, D)
            cmp_v_squeezed = cmp_v
            # Output per KV head: (B, T, H_kv, D)
            cmp_out_grouped = torch.einsum(
                'bhts,bshd->bthd',
                cmp_attn_grouped,
                cmp_v_squeezed,
            )
            # Repeat for all query heads: (B, T, H, D)
            cmp_out = cmp_out_grouped.repeat_interleave(heads_per_group, dim=2)
            cmp_out = cmp_out.reshape(B, T, self.n_embd)

            # ================================================================
            # Branch 2: Selection (fine-grained attention on top-k blocks)
            # ================================================================
            k = min(self.top_k_blocks, num_cmp)
            _, topk_idx = cmp_importance.topk(k, dim=-1)  # (B, T, k)

            # Gather selected blocks' keys and values
            # For each query position, select k blocks of individual tokens
            topk_starts = topk_idx * self.stride  # (B, T, k) — start position of each block
            block_offsets = torch.arange(self.block_size, device=x.device)  # (block_size,)
            selected_positions = topk_starts.unsqueeze(-1) + block_offsets.unsqueeze(0).unsqueeze(0)
            selected_positions = selected_positions.clamp(max=T - 1)  # (B, T, k, block_size)

            # Flatten block dimension for gather
            sel_pos_flat = selected_positions.reshape(B, T, k * self.block_size)  # (B, T, k*bs)

            # Gather keys and values at selected positions
            # slc_k: (B, T, n_kv_head, D), sel_pos_flat: (B, T, k*block_size)
            # Use advanced indexing: for each batch b, gather slc_k[b, sel_pos_flat[b], :, :]
            # This avoids the expensive loop+stack approach.
            B_idx = torch.arange(B, device=x.device).view(B, 1, 1)  # (B, 1, 1)
            # slc_k[B_idx, sel_pos_flat] -> (B, T, k*bs, n_kv_head, D)
            sel_k = slc_k[B_idx, sel_pos_flat]  # (B, T, k*bs, H_kv, D)
            sel_v = slc_v[B_idx, sel_pos_flat]

            # Padding mask for selected blocks
            sel_pad_mask = (sel_pos_flat < T).to(x.dtype)
            sel_k = sel_k * sel_pad_mask.unsqueeze(-1).unsqueeze(-1)

            # Reshape for SDPA: treat each query position as a separate "sequence"
            slc_q_rs = slc_q.reshape(B * T, self.n_head, 1, self.head_dim).to(x.dtype)
            slc_k_rs = sel_k.reshape(B * T, k * self.block_size, self.n_kv_head, self.head_dim).to(x.dtype)
            slc_v_rs = sel_v.reshape(B * T, k * self.block_size, self.n_kv_head, self.head_dim).to(x.dtype)
            slc_k_rs = slc_k_rs.permute(0, 2, 1, 3)  # (B*T, n_kv, sel_len, D)
            slc_v_rs = slc_v_rs.permute(0, 2, 1, 3)

            # Attention mask for padding
            slc_attn_mask = sel_pad_mask.reshape(B * T, 1, 1, k * self.block_size)
            slc_attn_mask = slc_attn_mask == 1

            # SDPA with GQA (query has n_head, kv has n_kv_head)
            sel_len = k * self.block_size
            slc_out = F.scaled_dot_product_attention(
                slc_q_rs, slc_k_rs, slc_v_rs,
                attn_mask=slc_attn_mask,
                enable_gqa=(self.n_head != self.n_kv_head),
            )  # (B*T, n_head, 1, D)
            slc_out = slc_out.reshape(B, T, self.n_head * self.head_dim)
        else:
            # Not enough tokens for compression — fall back to window-only
            slc_out = torch.zeros(B, T, self.n_embd, device=x.device, dtype=x.dtype)

        # ================================================================
        # Branch 3: Sliding window (local context)
        # ================================================================
        # Use standard SDPA with causal=True. The window is implicit:
        # at position t, attention naturally focuses on nearby tokens.
        # For explicit windowing, we could mask distant tokens, but at L=1024
        # with the model learning appropriate patterns, standard causal is fine.
        # The independent KV projections already prevent gradient interference.
        win_out = F.scaled_dot_product_attention(
            win_q.permute(0, 2, 1, 3),   # (B, H, T, D)
            win_k.permute(0, 2, 1, 3),
            win_v.permute(0, 2, 1, 3),
            is_causal=True,
            enable_gqa=(self.n_head != self.n_kv_head),
        )  # (B, H, T, D)
        win_out = win_out.permute(0, 2, 1, 3).reshape(B, T, self.n_embd)

        # ================================================================
        # Gated combination of three branches
        # ================================================================
        gate_logits = self.gate_mlp(x)  # (B, T, 3)
        gate = torch.sigmoid(gate_logits)  # (B, T, 3)
        y = gate[..., 0:1] * cmp_out + gate[..., 1:2] * slc_out + gate[..., 2:3] * win_out

        y = self.c_proj(y)
        return y


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
    def __init__(self, config, layer_idx, use_nsa=False):
        super().__init__()
        if use_nsa:
            self.attn = NSACausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


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
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.use_nsa = config.nsa_block_size > 0
        if self.use_nsa:
            print0(f"NSA enabled: block_size={config.nsa_block_size}, stride={config.nsa_stride or config.nsa_block_size}, top_k={config.nsa_top_k_blocks}, window={config.nsa_window_size}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, use_nsa=self.use_nsa) for layer_idx in range(config.n_layer)]),
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
            if isinstance(block.attn, NSACausalSelfAttention):
                # NSA: init all branch Q/K/V projections like standard Q/K/V
                for attr in ['cmp_q', 'cmp_k', 'cmp_v', 'slc_q', 'slc_k', 'slc_v', 'win_q', 'win_k', 'win_v']:
                    torch.nn.init.uniform_(getattr(block.attn, attr).weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                # Compression MLP: standard init
                for m in block.attn.cmp_mlp:
                    if isinstance(m, Linear):
                        torch.nn.init.uniform_(m.weight, -s, s)
                torch.nn.init.normal_(block.attn.cmp_pos.weight, mean=0.0, std=0.02)
                # Gate MLP: small init so gates start near sigmoid(0)=0.5
                torch.nn.init.uniform_(block.attn.gate_mlp[0].weight, -s * 0.01, s * 0.01)
                torch.nn.init.zeros_(block.attn.gate_mlp[0].bias)
                torch.nn.init.zeros_(block.attn.gate_mlp[2].weight)
                torch.nn.init.zeros_(block.attn.gate_mlp[2].bias)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
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

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

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
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

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
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
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
        for window, _ in self.window_sizes:
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
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
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
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
