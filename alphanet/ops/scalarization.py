"""Fused edge-frame scalarization + per-channel MLP (plan phase-3 operator B).

Reference computation (alphanet.py forward):
    p[e,f,h]   = sum_d edge_frame[e,d,f] * S[src[e],d,h]      (src = i or j)
    t          = (p0, p1^2, p2)
    scalar[e,h]= ( w2 . silu(W1 @ t + b1) + b2 + p0 ) / sqrt(H)

The reference materializes two [E,3,H] projections and, worse, two
[E,H,H/4] MLP hidden blocks that autograd keeps for backward — the largest
single allocation in the model. Here every intermediate lives in registers;
the backward recomputes them and emits grad_S (atomic scatter over nodes)
and grad_frame only. Weight gradients are intentionally not implemented:
the fused path is inference-only (training uses the reference path).

fp32 + CUDA only; the caller falls back to the reference otherwise.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _project(f0, f1, f2, f3, f4, f5, f6, f7, f8, s0, s1, s2):
    # p_f = sum_d frame[d,f] * S[d]; frame row-major [d,f]
    p0 = f0 * s0 + f3 * s1 + f6 * s2
    p1 = f1 * s0 + f4 * s1 + f7 * s2
    p2 = f2 * s0 + f5 * s1 + f8 * s2
    return p0, p1, p2


@triton.jit
def _mlp_out(p0, p1, p2, W1_ptr, b1_ptr, w2_ptr, b2_ptr, HQ, BLOCK_Q: tl.constexpr):
    # hidden[q,h] = silu(W1[q,0]*p0 + W1[q,1]*p1^2 + W1[q,2]*p2 + b1[q])
    q = tl.arange(0, BLOCK_Q)
    qm = q < HQ
    w1a = tl.load(W1_ptr + q * 3 + 0, mask=qm, other=0.0)
    w1b = tl.load(W1_ptr + q * 3 + 1, mask=qm, other=0.0)
    w1c = tl.load(W1_ptr + q * 3 + 2, mask=qm, other=0.0)
    b1 = tl.load(b1_ptr + q, mask=qm, other=0.0)
    w2 = tl.load(w2_ptr + q, mask=qm, other=0.0)
    t1 = p1 * p1
    pre = (w1a[:, None] * p0[None, :] + w1b[:, None] * t1[None, :]
           + w1c[:, None] * p2[None, :] + b1[:, None])
    hid = tl.where(qm[:, None], _silu(pre), 0.0)
    return tl.sum(hid * w2[:, None], axis=0) + tl.load(b2_ptr)


@triton.jit
def _fwd_kernel(S_ptr, frame_ptr, idx_i_ptr, idx_j_ptr,
                W1_ptr, b1_ptr, w2_ptr, b2_ptr, scale,
                out3_ptr, out4_ptr, H, HQ,
                BLOCK_H: tl.constexpr, BLOCK_Q: tl.constexpr):
    e = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    fbase = frame_ptr + e * 9
    f0 = tl.load(fbase + 0); f1 = tl.load(fbase + 1); f2 = tl.load(fbase + 2)
    f3 = tl.load(fbase + 3); f4 = tl.load(fbase + 4); f5 = tl.load(fbase + 5)
    f6 = tl.load(fbase + 6); f7 = tl.load(fbase + 7); f8 = tl.load(fbase + 8)

    i = tl.load(idx_i_ptr + e)
    s0 = tl.load(S_ptr + i * 3 * H + 0 * H + h, mask=hm, other=0.0)
    s1 = tl.load(S_ptr + i * 3 * H + 1 * H + h, mask=hm, other=0.0)
    s2 = tl.load(S_ptr + i * 3 * H + 2 * H + h, mask=hm, other=0.0)
    p0, p1, p2 = _project(f0, f1, f2, f3, f4, f5, f6, f7, f8, s0, s1, s2)
    out = _mlp_out(p0, p1, p2, W1_ptr, b1_ptr, w2_ptr, b2_ptr, HQ, BLOCK_Q)
    tl.store(out3_ptr + e * H + h, (out + p0) * scale, mask=hm)

    j = tl.load(idx_j_ptr + e)
    s0 = tl.load(S_ptr + j * 3 * H + 0 * H + h, mask=hm, other=0.0)
    s1 = tl.load(S_ptr + j * 3 * H + 1 * H + h, mask=hm, other=0.0)
    s2 = tl.load(S_ptr + j * 3 * H + 2 * H + h, mask=hm, other=0.0)
    p0, p1, p2 = _project(f0, f1, f2, f3, f4, f5, f6, f7, f8, s0, s1, s2)
    out = _mlp_out(p0, p1, p2, W1_ptr, b1_ptr, w2_ptr, b2_ptr, HQ, BLOCK_Q)
    tl.store(out4_ptr + e * H + h, (out + p0) * scale, mask=hm)


@triton.jit
def _bwd_dp(p0, p1, p2, g, W1_ptr, b1_ptr, w2_ptr, scale, HQ, BLOCK_Q: tl.constexpr):
    # recompute hidden, return dL/dp0, dL/dp1, dL/dp2 for one source
    q = tl.arange(0, BLOCK_Q)
    qm = q < HQ
    w1a = tl.load(W1_ptr + q * 3 + 0, mask=qm, other=0.0)
    w1b = tl.load(W1_ptr + q * 3 + 1, mask=qm, other=0.0)
    w1c = tl.load(W1_ptr + q * 3 + 2, mask=qm, other=0.0)
    b1 = tl.load(b1_ptr + q, mask=qm, other=0.0)
    w2 = tl.load(w2_ptr + q, mask=qm, other=0.0)
    t1 = p1 * p1
    pre = (w1a[:, None] * p0[None, :] + w1b[:, None] * t1[None, :]
           + w1c[:, None] * p2[None, :] + b1[:, None])
    sig = tl.sigmoid(pre)
    dsilu = sig * (1.0 + pre * (1.0 - sig))
    dout = g * scale                                   # [BLOCK_H]
    dpre = tl.where(qm[:, None], dsilu * (w2[:, None] * dout[None, :]), 0.0)
    dp0 = tl.sum(dpre * w1a[:, None], axis=0) + dout   # +dout: the "+p0" term
    dp1 = tl.sum(dpre * w1b[:, None], axis=0) * 2.0 * p1
    dp2 = tl.sum(dpre * w1c[:, None], axis=0)
    return dp0, dp1, dp2


@triton.jit
def _bwd_kernel(S_ptr, frame_ptr, idx_i_ptr, idx_j_ptr,
                W1_ptr, b1_ptr, w2_ptr, scale,
                g3_ptr, g4_ptr, gS_ptr, gframe_ptr, H, HQ,
                BLOCK_H: tl.constexpr, BLOCK_Q: tl.constexpr):
    e = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    fbase = frame_ptr + e * 9
    f0 = tl.load(fbase + 0); f1 = tl.load(fbase + 1); f2 = tl.load(fbase + 2)
    f3 = tl.load(fbase + 3); f4 = tl.load(fbase + 4); f5 = tl.load(fbase + 5)
    f6 = tl.load(fbase + 6); f7 = tl.load(fbase + 7); f8 = tl.load(fbase + 8)

    gf0 = 0.0; gf1 = 0.0; gf2 = 0.0; gf3 = 0.0; gf4 = 0.0
    gf5 = 0.0; gf6 = 0.0; gf7 = 0.0; gf8 = 0.0

    # ---- source i (scalar3) ----
    i = tl.load(idx_i_ptr + e)
    s0 = tl.load(S_ptr + i * 3 * H + 0 * H + h, mask=hm, other=0.0)
    s1 = tl.load(S_ptr + i * 3 * H + 1 * H + h, mask=hm, other=0.0)
    s2 = tl.load(S_ptr + i * 3 * H + 2 * H + h, mask=hm, other=0.0)
    p0, p1, p2 = _project(f0, f1, f2, f3, f4, f5, f6, f7, f8, s0, s1, s2)
    g = tl.load(g3_ptr + e * H + h, mask=hm, other=0.0)
    dp0, dp1, dp2 = _bwd_dp(p0, p1, p2, g, W1_ptr, b1_ptr, w2_ptr, scale, HQ, BLOCK_Q)
    tl.atomic_add(gS_ptr + i * 3 * H + 0 * H + h, f0 * dp0 + f1 * dp1 + f2 * dp2, mask=hm)
    tl.atomic_add(gS_ptr + i * 3 * H + 1 * H + h, f3 * dp0 + f4 * dp1 + f5 * dp2, mask=hm)
    tl.atomic_add(gS_ptr + i * 3 * H + 2 * H + h, f6 * dp0 + f7 * dp1 + f8 * dp2, mask=hm)
    gf0 += tl.sum(tl.where(hm, s0 * dp0, 0.0)); gf1 += tl.sum(tl.where(hm, s0 * dp1, 0.0)); gf2 += tl.sum(tl.where(hm, s0 * dp2, 0.0))
    gf3 += tl.sum(tl.where(hm, s1 * dp0, 0.0)); gf4 += tl.sum(tl.where(hm, s1 * dp1, 0.0)); gf5 += tl.sum(tl.where(hm, s1 * dp2, 0.0))
    gf6 += tl.sum(tl.where(hm, s2 * dp0, 0.0)); gf7 += tl.sum(tl.where(hm, s2 * dp1, 0.0)); gf8 += tl.sum(tl.where(hm, s2 * dp2, 0.0))

    # ---- source j (scalar4) ----
    j = tl.load(idx_j_ptr + e)
    s0 = tl.load(S_ptr + j * 3 * H + 0 * H + h, mask=hm, other=0.0)
    s1 = tl.load(S_ptr + j * 3 * H + 1 * H + h, mask=hm, other=0.0)
    s2 = tl.load(S_ptr + j * 3 * H + 2 * H + h, mask=hm, other=0.0)
    p0, p1, p2 = _project(f0, f1, f2, f3, f4, f5, f6, f7, f8, s0, s1, s2)
    g = tl.load(g4_ptr + e * H + h, mask=hm, other=0.0)
    dp0, dp1, dp2 = _bwd_dp(p0, p1, p2, g, W1_ptr, b1_ptr, w2_ptr, scale, HQ, BLOCK_Q)
    tl.atomic_add(gS_ptr + j * 3 * H + 0 * H + h, f0 * dp0 + f1 * dp1 + f2 * dp2, mask=hm)
    tl.atomic_add(gS_ptr + j * 3 * H + 1 * H + h, f3 * dp0 + f4 * dp1 + f5 * dp2, mask=hm)
    tl.atomic_add(gS_ptr + j * 3 * H + 2 * H + h, f6 * dp0 + f7 * dp1 + f8 * dp2, mask=hm)
    gf0 += tl.sum(tl.where(hm, s0 * dp0, 0.0)); gf1 += tl.sum(tl.where(hm, s0 * dp1, 0.0)); gf2 += tl.sum(tl.where(hm, s0 * dp2, 0.0))
    gf3 += tl.sum(tl.where(hm, s1 * dp0, 0.0)); gf4 += tl.sum(tl.where(hm, s1 * dp1, 0.0)); gf5 += tl.sum(tl.where(hm, s1 * dp2, 0.0))
    gf6 += tl.sum(tl.where(hm, s2 * dp0, 0.0)); gf7 += tl.sum(tl.where(hm, s2 * dp1, 0.0)); gf8 += tl.sum(tl.where(hm, s2 * dp2, 0.0))

    gbase = gframe_ptr + e * 9
    tl.atomic_add(gbase + 0, gf0); tl.atomic_add(gbase + 1, gf1); tl.atomic_add(gbase + 2, gf2)
    tl.atomic_add(gbase + 3, gf3); tl.atomic_add(gbase + 4, gf4); tl.atomic_add(gbase + 5, gf5)
    tl.atomic_add(gbase + 6, gf6); tl.atomic_add(gbase + 7, gf7); tl.atomic_add(gbase + 8, gf8)


_WARNED_WEIGHT_GRAD = False


def _blocks(H, HQ):
    BLOCK_H = 64 if H > 64 else triton.next_power_of_2(max(H, 16))
    BLOCK_Q = triton.next_power_of_2(max(HQ, 16))
    return BLOCK_H, BLOCK_Q


class _FusedScalarizationMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, frame, idx_i, idx_j, W1, b1, w2, b2, scale):
        E = frame.shape[0]
        H = S.shape[2]
        HQ = W1.shape[0]
        out3 = torch.empty(E, H, dtype=S.dtype, device=S.device)
        out4 = torch.empty(E, H, dtype=S.dtype, device=S.device)
        if E > 0:
            BLOCK_H, BLOCK_Q = _blocks(H, HQ)
            grid = (E, triton.cdiv(H, BLOCK_H))
            _fwd_kernel[grid](S, frame, idx_i, idx_j, W1, b1, w2,
                              b2, float(scale), out3, out4, H, HQ,
                              BLOCK_H=BLOCK_H, BLOCK_Q=BLOCK_Q)
        ctx.save_for_backward(S, frame, idx_i, idx_j, W1, b1, w2)
        ctx.scale = float(scale)
        return out3, out4

    @staticmethod
    def backward(ctx, g3, g4):
        if any(ctx.needs_input_grad[4:8]):
            # Inference calls autograd.grad(energy, positions): the engine may
            # still flag the (requires_grad) weights here even though their
            # gradients are pruned from the requested outputs. Returning None
            # is correct for that use. Actual training must not run through
            # the fused path (the model gates it on `not self.training`).
            global _WARNED_WEIGHT_GRAD
            if not _WARNED_WEIGHT_GRAD:
                import warnings
                warnings.warn(
                    "fused scalarization does not produce weight gradients; "
                    "train with use_fused_ops disabled")
                _WARNED_WEIGHT_GRAD = True
        S, frame, idx_i, idx_j, W1, b1, w2 = ctx.saved_tensors
        E = frame.shape[0]
        H = S.shape[2]
        HQ = W1.shape[0]
        gS = torch.zeros_like(S)
        gframe = torch.zeros_like(frame)
        if E > 0:
            BLOCK_H, BLOCK_Q = _blocks(H, HQ)
            grid = (E, triton.cdiv(H, BLOCK_H))
            _bwd_kernel[grid](S, frame, idx_i, idx_j, W1, b1, w2,
                              ctx.scale, g3.contiguous(), g4.contiguous(),
                              gS, gframe, H, HQ,
                              BLOCK_H=BLOCK_H, BLOCK_Q=BLOCK_Q)
        return gS, gframe, None, None, None, None, None, None, None


def fused_scalarization_mlp(S, edge_frame, idx_i, idx_j, lin, hidden_channels):
    """Drop-in fused replacement for the scalarization + lin MLP block.

    Args mirror the reference code: ``S`` is S_i_j [N,3,H], ``edge_frame``
    [E,3,3], ``idx_i``/``idx_j`` the edge target/source rows, ``lin`` the
    Sequential(Linear(3,H//4), SiLU, Linear(H//4,1)) module. Returns
    (scalar3, scalar4), each [E,H], already divided by sqrt(H).
    """
    W1 = lin[0].weight.contiguous()
    b1 = lin[0].bias.contiguous()
    w2 = lin[2].weight.reshape(-1).contiguous()
    b2 = (lin[2].bias.contiguous() if lin[2].bias is not None
          else torch.zeros(1, dtype=S.dtype, device=S.device))
    scale = 1.0 / math.sqrt(hidden_channels)
    return _FusedScalarizationMLP.apply(
        S.contiguous(), edge_frame.contiguous(),
        idx_i.contiguous(), idx_j.contiguous(), W1, b1, w2, b2, scale)
