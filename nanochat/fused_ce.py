"""
Fused linear + cross-entropy Triton kernel (Liger-style).

Fuses: matmul -> bf16 rounding -> softcap tanh -> cross-entropy NLL
into a single kernel pass that never materializes the full (BT, V) fp32 logits.

Forward: computes NLL and logsumexp per row, streaming over V tiles.
Backward: recomputes logits, computes softmax gradient, writes G buffer (bf16),
accumulates dX in fp32. dW computed as G^T @ X in torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    """Numerically stable tanh using exp. Handles large positive and negative values."""
    abs_x = tl.where(x >= 0, x, -x)
    e = tl.exp(2.0 * abs_x)
    result = 1.0 - 2.0 / (e + 1.0)
    return tl.where(x >= 0, result, -result)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel(
    X, W, targets,           # pointers
    NLL, LSE,                # output pointers
    BT, V,                   # runtime dimensions
    softcap,                 # float
    ignore_index,            # int
    stride_x_row, stride_x_col,
    stride_w_row, stride_w_col,
    D: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Forward kernel: one program per row block.

    Streams over V tiles computing:
      - online logsumexp (fp32)
      - target logit z_t (fp32)
    Stores per-row NLL and LSE.

    D is split into chunks of D_CHUNK to allow larger BLOCK_V within shared memory limits.
    When D_CHUNK == D, the inner D loop runs once (no overhead).
    """
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    v_col_offs = tl.arange(0, D_CHUNK)

    # Load targets for this row block
    t = tl.load(targets + row_offs, mask=row_offs < BT, other=ignore_index)
    valid = t != ignore_index

    # Running logsumexp state
    m = tl.full((BLOCK_B,), value=float('-inf'), dtype=tl.float32)
    l = tl.full((BLOCK_B,), value=0.0, dtype=tl.float32)
    zt = tl.full((BLOCK_B,), value=0.0, dtype=tl.float32)

    # Loop over vocab tiles
    for v0 in range(0, V, BLOCK_V):
        v_offs = v0 + tl.arange(0, BLOCK_V)

        # Accumulate z across D chunks
        z = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
        for d0 in range(0, D, D_CHUNK):
            d_offs = d0 + v_col_offs
            x_ptrs = X + row_offs[:, None] * stride_x_row + d_offs[None, :] * stride_x_col
            x_chunk = tl.load(x_ptrs, mask=(row_offs[:, None] < BT) & (d_offs[None, :] < D))
            w_ptrs = W + v_offs[:, None] * stride_w_row + d_offs[None, :] * stride_w_col
            w_chunk = tl.load(w_ptrs, mask=(v_offs[:, None] < V) & (d_offs[None, :] < D))
            z += tl.dot(x_chunk, tl.trans(w_chunk), out_dtype=tl.float32)

        # bf16 rounding (mimics reference precision)
        z = z.to(tl.bfloat16).to(tl.float32)

        # Softcap
        z = softcap * _tanh(z / softcap)

        # Accumulate target logit z_t
        is_target = (v_offs[None, :] == t[:, None])
        zt += tl.sum(tl.where(is_target, z, 0.0), axis=1)

        # Mask invalid rows
        z_masked = tl.where(valid[:, None], z, float('-inf'))

        # Online logsumexp (guard against NaN when both m and m_new are -inf)
        m_new = tl.maximum(m, tl.max(z_masked, axis=1))
        safe_scale = tl.where(m_new == float('-inf'), 1.0, tl.exp(m - m_new))
        l = l * safe_scale
        exp_contrib = tl.exp(z_masked - m_new[:, None])
        exp_contrib = tl.where(m_new[:, None] == float('-inf'), 0.0, exp_contrib)
        l = l + tl.sum(exp_contrib, axis=1)
        m = m_new

    # Finalize
    lse = tl.log(l) + m
    lse = tl.where(valid, lse, 0.0)
    nll = tl.where(valid, lse - zt, 0.0)

    tl.store(NLL + row_offs, nll, mask=row_offs < BT)
    tl.store(LSE + row_offs, lse, mask=row_offs < BT)


@triton.jit
def _bwd_kernel(
    X, W, targets, LSE, DY,  # inputs
    DX, G,                    # outputs
    BT, V,                    # runtime dimensions
    softcap,                  # float
    ignore_index,             # int
    stride_x_row, stride_x_col,
    stride_w_row, stride_w_col,
    stride_g_row,
    stride_dx_row, stride_dx_col,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Backward kernel: one program per row block.

    Recomputes logits tile-by-tile, computes gradient:
      g = (softmax(z) - one_hot(t)) * softcap_deriv * dy
    Writes G tiles (bf16) and accumulates dX (fp32).

    Cannot split D in backward because we need full z for softmax.
    Uses smaller BLOCK_B to fit dX_acc in shared memory.
    """
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    col_offs = tl.arange(0, D)

    # Load x block (keep bf16 for tl.dot)
    x_ptrs = X + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    x_block = tl.load(x_ptrs, mask=row_offs[:, None] < BT)

    # Load targets, lse, dy
    t = tl.load(targets + row_offs, mask=row_offs < BT, other=ignore_index)
    lse = tl.load(LSE + row_offs, mask=row_offs < BT)
    dy = tl.load(DY + row_offs, mask=row_offs < BT)

    valid = t != ignore_index
    dy = tl.where(valid, dy, 0.0)

    # dX accumulator in fp32
    dX_acc = tl.zeros((BLOCK_B, D), dtype=tl.float32)

    for v0 in range(0, V, BLOCK_V):
        v_offs = v0 + tl.arange(0, BLOCK_V)

        w_ptrs = W + v_offs[:, None] * stride_w_row + col_offs[None, :] * stride_w_col
        w_tile = tl.load(w_ptrs, mask=v_offs[:, None] < V)

        # Recompute z exactly as forward
        z = tl.dot(x_block, tl.trans(w_tile), out_dtype=tl.float32)
        z = z.to(tl.bfloat16).to(tl.float32)
        z = softcap * _tanh(z / softcap)

        # Softmax probabilities
        p = tl.exp(z - lse[:, None])
        p = tl.where(valid[:, None], p, 0.0)

        # Cross-entropy gradient: p - one_hot(t)
        is_target = (v_offs[None, :] == t[:, None])
        g = tl.where(is_target, p - 1.0, p)

        # Softcap derivative: d(softcap*tanh(u/softcap))/du = 1 - (z/softcap)^2
        g = g * (1.0 - (z / softcap) * (z / softcap))

        # Scale by upstream grad
        g = g * dy[:, None]

        # Store G tile as bf16
        g_ptrs = G + row_offs[:, None] * stride_g_row + v_offs[None, :]
        tl.store(g_ptrs, g.to(tl.bfloat16), mask=(row_offs[:, None] < BT) & (v_offs[None, :] < V))

        # Accumulate dX: g_bf16 @ w_tile (matches reference bf16 GEMM precision)
        dX_acc += tl.dot(g.to(tl.bfloat16), w_tile, out_dtype=tl.float32)

    # Store dX as bf16
    dx_ptrs = DX + row_offs[:, None] * stride_dx_row + col_offs[None, :] * stride_dx_col
    tl.store(dx_ptrs, dX_acc.to(tl.bfloat16), mask=row_offs[:, None] < BT)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------

class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, softcap, ignore_index, reduction):
        BT, D = x.shape
        V = weight.shape[0]
        assert x.is_contiguous()
        assert weight.is_contiguous()
        assert targets.is_contiguous()
        assert D % 128 == 0 and V % 64 == 0 and BT % 64 == 0

        # Best config from benchmarking on RTX 3060 Ti (sm86):
        # Forward: BLOCK_B=64, BLOCK_V=64, D_CHUNK=D (no chunking), num_warps=4 → ~10ms
        BLOCK_B = 64
        BLOCK_V = 64
        D_CHUNK = D  # D=256 fits in shared memory with BLOCK_B=64, BLOCK_V=64

        nll = torch.empty(BT, dtype=torch.float32, device=x.device)
        lse = torch.empty(BT, dtype=torch.float32, device=x.device)

        grid = (triton.cdiv(BT, BLOCK_B),)
        _fwd_kernel[grid](
            x, weight, targets,
            nll, lse,
            BT, V,
            softcap, ignore_index,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            D=D, D_CHUNK=D_CHUNK, BLOCK_B=BLOCK_B, BLOCK_V=BLOCK_V,
            num_warps=4,
        )

        valid_mask = targets != ignore_index
        num_valid = valid_mask.sum()
        if reduction == 'mean':
            loss = nll.sum() / num_valid.clamp(min=1)
        else:
            loss = nll.sum()

        ctx.save_for_backward(x, weight, targets, lse)
        ctx.softcap = softcap
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.num_valid = num_valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, targets, lse = ctx.saved_tensors
        BT, D = x.shape
        V = weight.shape[0]

        # Best config from benchmarking: BLOCK_B=32, BLOCK_V=64, num_warps=4 → ~17.6ms
        # Smaller BLOCK_B needed to fit dX_acc (32*256*4=32KB) in shared memory
        BLOCK_B = 32
        BLOCK_V = 64

        if ctx.reduction == 'mean':
            dy = torch.where(targets != ctx.ignore_index,
                             grad_output / ctx.num_valid.clamp(min=1),
                             torch.zeros(1, device=grad_output.device, dtype=grad_output.dtype))
        else:
            dy = torch.where(targets != ctx.ignore_index,
                             grad_output,
                             torch.zeros(1, device=grad_output.device, dtype=grad_output.dtype))
        dy = dy.to(torch.float32)

        G = torch.empty(BT, V, dtype=torch.bfloat16, device=x.device)
        dx = torch.empty(BT, D, dtype=torch.bfloat16, device=x.device)

        assert BT % BLOCK_B == 0 and V % BLOCK_V == 0
        grid = (triton.cdiv(BT, BLOCK_B),)
        _bwd_kernel[grid](
            x, weight, targets, lse, dy,
            dx, G,
            BT, V,
            ctx.softcap, ctx.ignore_index,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            G.stride(0),
            dx.stride(0), dx.stride(1),
            D=D, BLOCK_B=BLOCK_B, BLOCK_V=BLOCK_V,
            num_warps=4,
        )

        dW = torch.mm(G.t(), x).float() if ctx.needs_input_grad[1] else None

        return dx, dW, None, None, None, None


def fused_linear_ce(x, weight, targets, softcap=15.0, ignore_index=-1, reduction='mean'):
    """
    Fused linear + softcap + cross-entropy loss.

    Args:
        x: (BT, D) bf16 cuda
        weight: (V, D) bf16 or fp32 cuda (cast to bf16 internally)
        targets: (BT,) int64 cuda, entries == ignore_index are skipped
        softcap: float, logit softcapping value
        ignore_index: int, target value to ignore
        reduction: 'mean' or 'sum'

    Returns:
        scalar loss
    """
    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)
    return _FusedLinearCE.apply(x, weight, targets, softcap, ignore_index, reduction)
