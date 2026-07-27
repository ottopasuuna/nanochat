"""
Mamba-2 (SSD) mixer module for hybrid Mamba/Attention models.

Implements the chunked SSD (State Space Duality) algorithm from:
    "Transformers are SSMs: Generalized Models and Efficient Algorithms
     Through Structured State Space Duality" (Dao & Gu, 2024)

Faithfully follows the reference implementation in ssd_minimal.py from:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py

Pure PyTorch, no external dependencies (no mamba-ssm, no causal-conv1d).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE


class Mamba2Mixer(nn.Module):
    """
    Mamba-2 block using the SSD (State Space Duality) formulation.

    The core insight of Mamba-2 is that the selective SSM can be written as a
    semiseparable matrix multiplication, which decomposes into:
    1. Intra-chunk: attention-like matmul with causal decay masking
    2. Inter-chunk: sequential recurrence over chunk states (tiny loop)

    This maps naturally to matmul-friendly operations, enabling 2-8x speedup
    over Mamba-1's scan-based approach while using tensor cores.
    """

    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64,
                 ngroups=1, chunk_size=256, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        # in_proj: x -> [z | xBC | dt]
        # z: d_inner (gating)
        # xBC: d_inner + 2 * ngroups * d_state (x for SSM, B and C for state mixing)
        # dt: nheads (timestep for discretization)
        d_xBC = self.d_inner + 2 * ngroups * d_state
        d_in_proj = self.d_inner + d_xBC + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # Causal depthwise conv on xBC (pure PyTorch, no causal-conv1d dep)
        self.conv1d = nn.Conv1d(
            in_channels=d_xBC,
            out_channels=d_xBC,
            bias=True,
            kernel_size=d_conv,
            groups=d_xBC,
            padding=d_conv - 1,
        )
        self._d_xBC = d_xBC

        # SSM parameters
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Grouped RMSNorm (gated: applied to SSM output before gating by z)
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _run_conv1d(self, x):
        """Causal conv1d with dtype-cast weights (matching nanochat Linear convention).
        Input: (B, d_xBC, L). Output: (B, d_xBC, L) causal."""
        w = self.conv1d.weight.to(dtype=x.dtype)
        b = self.conv1d.bias.to(dtype=x.dtype) if self.conv1d.bias is not None else None
        x = F.conv1d(x, w, b, padding=self.d_conv - 1, groups=self._d_xBC)
        return x[:, :, :x.shape[-1] - self.d_conv + 1]  # causal slice

    def segsum(self, x):
        """Stable segment sum calculation for chunk-wise decay.

        Given x of shape (..., L), computes a (..., L, L) matrix where
        result[..., i, j] = sum(x[..., j+1:i+1]) for i >= j, -inf otherwise.
        This is used to compute cumulative decay within a chunk.

        Faithful to reference ssd_minimal.py segsum().
        All computation in fp32 for numerical stability.
        """
        T = x.size(-1)
        # Expand to pairwise: (..., L) -> (..., L, L) via broadcast
        x = x.unsqueeze(-1).expand(*x.shape, T)
        # Mask strictly lower triangle (positions above diagonal get 0)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        # Cumulative sum down columns gives segment sums
        x_segsum = torch.cumsum(x, dim=-2)
        # Mask to causal (lower triangular including diagonal) and fill with -inf
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
        return x_segsum

    def _chunked_ssd_forward(self, x, dt, A, B, C):
        """
        Chunked SSD forward pass (training/prefill).

        Faithful implementation of the SSD algorithm (Listing 1 from the Mamba-2 paper),
        following the reference ssd_minimal.py from state-spaces/mamba.

        The discrete SSM is: h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
                              y_t = C_t · h_t

        In SSD form, we pass X_discrete = x*dt and A_discrete = A*dt.

        Args:
            x: (B, L, H, P) - input projected to head space (raw, NOT pre-scaled by dt)
            dt: (B, L, H) - discretization timestep (already softplus'd, positive)
            A: (H,) - negative continuous-time state transition rates
            B: (B, L, G, N) - input-dependent state mixing (left)
            C: (B, L, G, N) - state-to-output mixing (right)

        Returns:
            y: (B, L, H, P) - SSM output (C*h terms only, D*x added by caller)
            final_state: (B, H, P, N) - final SSM state (for KV cache)
        """
        B_batch, L, H, P = x.shape
        G = self.ngroups
        N = self.d_state
        CS = self.chunk_size

        # Pad sequence length to multiple of chunk_size
        n_chunks = (L + CS - 1) // CS
        padded_len = n_chunks * CS
        if padded_len != L:
            pad_len = padded_len - L
            x = F.pad(x, (0, 0, 0, pad_len))    # (B, padded_len, H, P)
            dt = F.pad(dt, (0, pad_len))          # (B, padded_len, H)
            B = F.pad(B, (0, 0, 0, pad_len))     # (B, padded_len, G, N)
            C = F.pad(C, (0, 0, 0, pad_len))     # (B, padded_len, G, N)

        # Reshape into chunks: (B, L, ...) -> (B, n_chunks, CS, ...)
        # Following reference: b (c l) ... -> b c l ...
        x = x.reshape(B_batch, n_chunks, CS, H, P)      # (b, c, l, h, p)
        dt = dt.reshape(B_batch, n_chunks, CS, H)         # (b, c, l, h)
        B = B.reshape(B_batch, n_chunks, CS, G, N)        # (b, c, l, g, n)
        C = C.reshape(B_batch, n_chunks, CS, G, N)        # (b, c, l, g, n)

        # Expand B/C for head grouping: (b, c, l, g, n) -> (b, c, l, h, n)
        if G == 1:
            B_expanded = B.expand(-1, -1, -1, H, -1)  # (b, c, l, h, n)
            C_expanded = C.expand(-1, -1, -1, H, -1)
        else:
            heads_per_group = H // G
            B_expanded = B.repeat_interleave(heads_per_group, dim=3)
            C_expanded = C.repeat_interleave(heads_per_group, dim=3)

        # Compute discrete-form quantities (matching reference ssd_minimal_discrete convention):
        # A_discrete = dt * A (for decay), X_discrete = x * dt (for input)
        # dtA layout: (b, c, l, h) -> transpose to (b, h, c, l) to match reference
        dtA = dt * A.float().view(1, 1, 1, H)   # (b, c, l, h)
        dtA = dtA.permute(0, 3, 1, 2)            # (b, h, c, l) — matches reference "bhcl" layout
        A_cumsum = torch.cumsum(dtA, dim=-1)     # (b, h, c, l)

        # Pre-multiply x by dt (discrete form): X_discrete = x * dt
        x_scaled = (x * dt.unsqueeze(-1)).to(COMPUTE_DTYPE)  # (b, c, l, h, p)

        # --- 1. Intra-chunk: diagonal blocks ---
        # L = exp(segsum(A_discrete)) where A_discrete has time dim last
        # segsum on (b, h, c, l) operates on last dim l, gives (b, h, c, l, l)
        L_decay = torch.exp(self.segsum(dtA))  # (b, h, c, l, l) in fp32

        # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X_scaled)
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_expanded.to(COMPUTE_DTYPE),
            B_expanded.to(COMPUTE_DTYPE),
            L_decay.to(COMPUTE_DTYPE),
            x_scaled,
        )

        # --- 2. Compute state for each intra-chunk ---
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        # decay_states: how much each position decays to end of chunk
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (b, h, c, l)
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_expanded.to(COMPUTE_DTYPE),
            decay_states.to(COMPUTE_DTYPE),
            x_scaled,
        )

        # --- 3. Inter-chunk recurrence (sequential scan over chunk states) ---
        # (middle term of factorization of off-diagonal blocks; A terms)
        initial_states = torch.zeros(B_batch, 1, H, P, N, device=x.device, dtype=states.dtype)
        states = torch.cat([initial_states, states], dim=1)  # (b, c+1, h, p, n)

        # Chunk-to-chunk decay from A at chunk boundaries
        # A_cumsum[:, :, :, -1] is (b, h, c) — cumulative A at end of each chunk
        # Pad with 0 at start for initial state, then segsum
        chunk_boundary_A = F.pad(A_cumsum[:, :, :, -1], (1, 0))  # (b, h, c+1)
        decay_chunk = torch.exp(self.segsum(chunk_boundary_A.float()))  # (b, h, c+1, c+1)

        # Apply recurrence: new_states[c] = sum_{z<=c} decay_chunk[c,z] * states[z]
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states.float())
        states = new_states[:, :-1]    # (b, c, h, p, n) — states at chunk starts
        final_state = new_states[:, -1]  # (b, h, p, n) — state after last chunk

        # --- 4. State -> output conversion per chunk ---
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
        Y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C_expanded.to(COMPUTE_DTYPE),
            states.to(COMPUTE_DTYPE),
            state_decay_out.to(COMPUTE_DTYPE),
        )

        # Combine intra-chunk and inter-chunk, unpad
        Y = (Y_diag + Y_off).float()  # (b, c, l, h, p)
        Y = Y.reshape(B_batch, -1, H, P)[:, :L, :, :]  # (B, L, H, P)

        return Y, final_state

    def _forward_naive(self, x, dt, A, B, C):
        """
        Naive sequential recurrence (reference implementation for validation).

        Computes the same SSM as _chunked_ssd_forward but via explicit loop:
        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
        y_t = C_t · h_t

        Much slower but simpler and easier to verify correctness.
        """
        B_batch, L, H, P = x.shape
        N = self.d_state
        G = self.ngroups

        # Expand B/C for head grouping
        if G == 1:
            B_expanded = B.expand(-1, -1, H, -1)  # (B, L, H, N)
            C_expanded = C.expand(-1, -1, H, -1)
        else:
            heads_per_group = H // G
            B_expanded = B.repeat_interleave(heads_per_group, dim=2)
            C_expanded = C.repeat_interleave(heads_per_group, dim=2)

        h = torch.zeros(B_batch, H, P, N, device=x.device, dtype=torch.float32)
        A_f = A.float()  # (H,)

        outputs = []
        for t in range(L):
            dt_t = dt[:, t, :].float()        # (B, H)
            x_t = x[:, t, :, :].float()       # (B, H, P)
            B_t = B_expanded[:, t, :, :].float()  # (B, H, N)
            C_t = C_expanded[:, t, :, :].float()  # (B, H, N)

            # Decay: exp(dt * A) where A is negative
            dA = torch.exp(dt_t.unsqueeze(-1).unsqueeze(-1) * A_f.view(1, H, 1, 1))  # (B, H, 1, 1)

            # Input: dt * B * x
            dBx = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(2) * x_t.unsqueeze(-1)  # (B, H, P, N)

            # Recurrence
            h = dA * h + dBx

            # Output: y = C · h
            y_t = (h * C_t.unsqueeze(2)).sum(dim=-1)  # (B, H, P)
            outputs.append(y_t)

        Y = torch.stack(outputs, dim=1)  # (B, L, H, P)
        return Y, h

    def step(self, x_t, conv_state, ssm_state):
        """
        Single-token recurrent decode step.

        For use during autoregressive inference (T=1).
        Updates conv_state and ssm_state and returns the output.

        Args:
            x_t: (B, 1, d_model) - single input token
            conv_state: (B, d_xBC, d_conv) - conv shift register
            ssm_state: (B, H, P, N) - SSM hidden state

        Returns:
            y: (B, 1, d_model) - output for this token
            conv_state: updated conv state
            ssm_state: updated ssm state
        """
        B = x_t.size(0)

        # Project input
        zxbcdt = F.linear(x_t.squeeze(1), self.in_proj.weight.to(dtype=x_t.dtype))  # (B, d_in_proj)
        d_xBC = self.d_inner + 2 * self.ngroups * self.d_state
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, d_xBC, self.nheads], dim=-1)

        # Conv1d step: shift register + new input
        # Shift conv_state right and insert new xBC at the end
        conv_state_new = torch.roll(conv_state, shifts=-1, dims=-1).clone()
        conv_state_new[:, :, -1] = xBC
        # Apply conv weights (depthwise: groups=d_xBC), cast to input dtype
        conv_weight = self.conv1d.weight.squeeze(1).to(dtype=xBC.dtype)  # (d_xBC, d_conv)
        xBC = (conv_state_new * conv_weight).sum(dim=-1)  # (B, d_xBC)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias.to(dtype=xBC.dtype)
        xBC = F.silu(xBC)

        # Split into x, B_ssm, C_ssm
        x, B_ssm, C_ssm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = x.view(B, self.nheads, self.headdim)  # (B, H, P)

        # Discretize
        A = -torch.exp(self.A_log.float())  # (H,)
        dt_soft = F.softplus(dt.float() + self.dt_bias.float())  # (B, H)

        # SSM step: h = exp(dt*A) * h + dt * B * x
        dA = torch.exp(dt_soft * A).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # B_ssm: (B, G*N) -> (B, H, N) via broadcast
        if self.ngroups == 1:
            B_ssm = B_ssm.view(B, 1, self.d_state).expand(-1, self.nheads, -1)  # (B, H, N)
            C_ssm = C_ssm.view(B, 1, self.d_state).expand(-1, self.nheads, -1)
        else:
            heads_per_group = self.nheads // self.ngroups
            B_ssm = B_ssm.view(B, self.ngroups, self.d_state).repeat_interleave(heads_per_group, dim=1)
            C_ssm = C_ssm.view(B, self.ngroups, self.d_state).repeat_interleave(heads_per_group, dim=1)

        # dBx = dt * B * x: (B, H, P, N)
        dBx = dt_soft.unsqueeze(-1).unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)

        # Update state
        ssm_state_new = dA * ssm_state.float() + dBx

        # Output: y = C @ h + D * x
        y = (ssm_state_new * C_ssm.unsqueeze(2)).sum(dim=-1)  # (B, H, P)
        y = y + self.D.view(1, -1, 1).float() * x.float()
        y = y.to(x_t.dtype)

        # Grouped RMSNorm + gate + output proj
        y = y.reshape(B, self.d_inner)
        y = self._grouped_rmsnorm(y)
        z = F.silu(z)
        y = y * z
        y = F.linear(y, self.out_proj.weight.to(dtype=y.dtype))

        return y.unsqueeze(1), conv_state_new, ssm_state_new

    def _grouped_rmsnorm(self, x):
        """Grouped RMSNorm: normalize within groups of size headdim.
        Computes in fp32 for stability, casts back to input dtype."""
        input_dtype = x.dtype
        shape = x.shape
        x = x.view(*shape[:-1], self.nheads, self.headdim)
        # RMS norm per head (fp32 for stability)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        x = x.float() * rms
        # Apply learnable weight per head
        w = self.norm_weight.view(self.nheads, self.headdim)
        x = x * w.float()
        return x.view(shape).to(input_dtype)

    def forward(self, x, kv_cache=None):
        """
        Forward pass.

        Args:
            x: (B, L, d_model) - input (already normed by caller)
            kv_cache: optional KVCache for inference

        Returns:
            y: (B, L, d_model) - output
        """
        B, L, _ = x.shape

        # KV cache path: dispatch to step() for T=1, chunked for T>1
        if kv_cache is not None:
            if L == 1:
                # Decode: single token, use recurrent step
                return self._kv_cache_step(x, kv_cache)
            else:
                # Prefill: chunked forward + store final states
                return self._kv_cache_prefill(x, kv_cache)

        # 1. Input projection and split (cast weights like nanochat Linear)
        zxbcdt = F.linear(x, self.in_proj.weight.to(dtype=x.dtype))  # (B, L, d_in_proj)
        d_xBC = self.d_inner + 2 * self.ngroups * self.d_state
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, d_xBC, self.nheads], dim=-1)

        # 2. Causal conv1d on xBC (dtype-cast like nanochat Linear)
        xBC = xBC.transpose(1, 2)  # (B, d_xBC, L)
        xBC = self._run_conv1d(xBC)  # (B, d_xBC, L) causal
        xBC = xBC.transpose(1, 2)  # (B, L, d_xBC)
        xBC = F.silu(xBC)

        # 3. Split xBC into x, B_ssm, C_ssm
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # Reshape for SSM
        x_ssm = x_ssm.view(B, L, self.nheads, self.headdim)  # (B, L, H, P)
        B_ssm = B_ssm.view(B, L, self.ngroups, self.d_state)  # (B, L, G, N)
        C_ssm = C_ssm.view(B, L, self.ngroups, self.d_state)  # (B, L, G, N)

        # 4. Discretization
        # A is negative (decay rates): A = -exp(A_log)
        A = -torch.exp(self.A_log.float())  # (H,)
        # dt must be positive: apply softplus with bias
        dt = F.softplus(dt.float() + self.dt_bias.float())  # (B, L, H)

        # 5. Chunked SSD forward (training path)
        y_ssm, _ = self._chunked_ssd_forward(x_ssm, dt, A, B_ssm, C_ssm)

        # 6. D (skip) connection
        y_ssm = y_ssm + self.D.view(1, 1, -1, 1).float() * x_ssm.float()
        y_ssm = y_ssm.to(x.dtype)

        # 7. Grouped RMSNorm + SiLU gate + output projection
        y = y_ssm.reshape(B, L, self.d_inner)
        y = self._grouped_rmsnorm(y)
        z = F.silu(z)
        y = y * z
        y = F.linear(y, self.out_proj.weight.to(dtype=y.dtype))

        return y

    def _kv_cache_prefill(self, x, kv_cache):
        """Prefill: chunked forward + store final SSM and conv states."""
        B, L, _ = x.shape
        assert self.layer_idx is not None, "layer_idx must be set for KV cache inference"
        layer_idx = self.layer_idx

        # Standard forward path
        zxbcdt = F.linear(x, self.in_proj.weight.to(dtype=x.dtype))
        d_xBC = self.d_inner + 2 * self.ngroups * self.d_state
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, d_xBC, self.nheads], dim=-1)

        xBC = xBC.transpose(1, 2)
        xBC = self._run_conv1d(xBC).transpose(1, 2)
        xBC = F.silu(xBC)

        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_ssm = x_ssm.view(B, L, self.nheads, self.headdim)
        B_ssm = B_ssm.view(B, L, self.ngroups, self.d_state)
        C_ssm = C_ssm.view(B, L, self.ngroups, self.d_state)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt.float() + self.dt_bias.float())

        y_ssm, final_state = self._chunked_ssd_forward(x_ssm, dt, A, B_ssm, C_ssm)

        # Store final states for subsequent decode steps
        # Conv state: last d_conv tokens of xBC (B, d_xBC, d_conv)
        conv_state = F.pad(xBC.transpose(1, 2), (self.d_conv - L, 0))[:, :, :self.d_conv]
        kv_cache.mamba_states[layer_idx] = (conv_state, final_state)

        # D (skip) connection + norm + gate + output proj
        y_ssm = y_ssm + self.D.view(1, 1, -1, 1).float() * x_ssm.float()
        y_ssm = y_ssm.to(x.dtype)
        y = y_ssm.reshape(B, L, self.d_inner)
        y = self._grouped_rmsnorm(y)
        z = F.silu(z)
        y = y * z
        y = F.linear(y, self.out_proj.weight.to(dtype=y.dtype))
        return y

    def _kv_cache_step(self, x, kv_cache):
        """Decode: single token using recurrent step()."""
        B, L, _ = x.shape
        assert L == 1
        assert self.layer_idx is not None, "layer_idx must be set for KV cache inference"
        layer_idx = self.layer_idx

        # Get cached states
        if layer_idx not in kv_cache.mamba_states:
            # Initialize states if not yet created (first decode without prefill)
            d_xBC = self.d_inner + 2 * self.ngroups * self.d_state
            conv_state = torch.zeros(B, d_xBC, self.d_conv, device=x.device, dtype=x.dtype)
            ssm_state = torch.zeros(B, self.nheads, self.headdim, self.d_state, device=x.device, dtype=torch.float32)
        else:
            conv_state, ssm_state = kv_cache.mamba_states[layer_idx]

        y, conv_state, ssm_state = self.step(x, conv_state, ssm_state)
        # Update states in cache
        kv_cache.mamba_states[layer_idx] = (conv_state, ssm_state)
        return y

    def init_weights(self):
        """
        Initialize Mamba-2 weights following the Mamba-2 paper conventions.

        - A_log: log(1..nheads) (ensures well-conditioned decay)
        - dt_bias: inverse-softplus of uniform(log(1e-3), log(0.1)) (positive timesteps)
        - D: ones (skip connection)
        - in_proj: uniform ±1/sqrt(d_model)
        - out_proj: zeros (nanochat convention)
        - conv1d: default PyTorch init
        - norm_weight: ones (already set in Parameter init)
        """
        n_embd = self.d_model
        s = 3**0.5 * n_embd**-0.5

        # A_log: log(1..nheads) -- values from 1 to nheads, then log
        # This gives a range of decay rates that cover both fast and slow dynamics
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log.data.copy_(torch.log(A))

        # dt_bias: inverse-softplus of uniform samples in [1e-3, 0.1]
        # This gives reasonable initial discretization timesteps
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

        # D: ones
        self.D.data.fill_(1.0)

        # norm_weight: ones
        self.norm_weight.data.fill_(1.0)

        # in_proj: uniform ±s
        torch.nn.init.uniform_(self.in_proj.weight, -s, s)

        # out_proj: zeros (nanochat convention for output projections)
        torch.nn.init.zeros_(self.out_proj.weight)

    def mamba_state_bytes(self, batch_size=1, dtype=None):
        """Bytes for storing SSM + conv states for inference."""
        if dtype is None:
            dtype = COMPUTE_DTYPE
        d_xBC = self.d_inner + 2 * self.ngroups * self.d_state
        conv_bytes = batch_size * d_xBC * self.d_conv * dtype.itemsize
        ssm_bytes = batch_size * self.nheads * self.headdim * self.d_state * dtype.itemsize
        return conv_bytes + ssm_bytes
