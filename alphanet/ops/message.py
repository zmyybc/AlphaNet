"""Fused EquiMessagePassing message+aggregate (plan phase-3 operator A).

Reference (EquiMessagePassing.message/aggregate, real-arithmetic form):
    x1, xh2, xh3 = split(xh[j] * rbfh, H)          # per edge
    (qr | qi)    = scale(x1)                       # edge GEMM, stays in cuBLAS
    conv         = q @ K[C:] + bias(K[:C])         # complex via real pairs
    ker          = (conv*a).sum() * u + conv.sum() * v      # rank-1 collapse
    msg_kernel   = atan2(fc(ker_i), fc(ker_r))     # [E, chi1]
    agg          = cat(msg_kernel, x1) * mask      # -> scatter_add -> dx
    vec_msg      = (vec[j]*xh2/sqrt3 + xh3 (x) r_ij)/sqrt(H) * mask
                                                    # -> scatter_add -> dvec

The wrapper keeps the three cuBLAS GEMMs (scale, diagonal MLP) outside; the
kernel fuses everything per edge in registers and accumulates straight into
the node outputs, so the [E,3H] gathers, [E,3,H] vector messages and
[E,chi1+H] message tensors (and their autograd copies) never exist.
Backward recomputes the chain per edge and emits grads for the tensor
inputs only (weights: inference-only, model gates on `not self.training`).

fp32 + CUDA only.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(xh_ptr, vec_ptr, rbfh_ptr, x1_ptr, qr_ptr, qi_ptr, a_ptr,
                r_ptr, mask_ptr, idx_i_ptr, idx_j_ptr,
                Kr_ptr, Ki_ptr, u_ptr, v_ptr, fcw_ptr, fcb_ptr,
                dx_ptr, dvec_ptr,
                H, HCHI, CHI1, CHI2, C, inv_sqrt_3, inv_sqrt_h,
                HAS_MASK: tl.constexpr,
                BLOCK_H: tl.constexpr, BLOCK_HCHI: tl.constexpr,
                BLOCK_C1: tl.constexpr, BLOCK_C2: tl.constexpr):
    e = tl.program_id(0)
    i = tl.load(idx_i_ptr + e)
    j = tl.load(idx_j_ptr + e)
    if HAS_MASK:
        m = tl.load(mask_ptr + e)
    else:
        m = 1.0

    # ---- MPS contraction: conv[k] = sum_c qr/qi[c] * K[C+c, k] + bias_k ----
    c2 = tl.arange(0, BLOCK_C2)
    c2m = c2 < CHI2
    cc = tl.arange(0, BLOCK_HCHI)
    ccm = cc < HCHI
    qr = tl.load(qr_ptr + e * HCHI + cc, mask=ccm, other=0.0)
    qi = tl.load(qi_ptr + e * HCHI + cc, mask=ccm, other=0.0)
    # K rows: first C rows are the "ones" bias, then HCHI rows
    kr_bias = tl.zeros([BLOCK_C2], dtype=tl.float32)
    ki_bias = tl.zeros([BLOCK_C2], dtype=tl.float32)
    for cidx in range(0, C):
        kr_bias += tl.load(Kr_ptr + cidx * CHI2 + c2, mask=c2m, other=0.0)
        ki_bias += tl.load(Ki_ptr + cidx * CHI2 + c2, mask=c2m, other=0.0)
    Krm = tl.load(Kr_ptr + (C + cc[:, None]) * CHI2 + c2[None, :],
                  mask=ccm[:, None] & c2m[None, :], other=0.0)
    Kim = tl.load(Ki_ptr + (C + cc[:, None]) * CHI2 + c2[None, :],
                  mask=ccm[:, None] & c2m[None, :], other=0.0)
    conv_r = tl.sum(qr[:, None] * Krm - qi[:, None] * Kim, axis=0) + kr_bias
    conv_i = tl.sum(qr[:, None] * Kim + qi[:, None] * Krm, axis=0) + ki_bias

    # ---- rank-1 dia collapse + fc_mps + atan2 ----
    a = tl.load(a_ptr + e * CHI2 + c2, mask=c2m, other=0.0)
    ca_r = tl.sum(conv_r * a)
    ca_i = tl.sum(conv_i * a)
    cs_r = tl.sum(tl.where(c2m, conv_r, 0.0))
    cs_i = tl.sum(tl.where(c2m, conv_i, 0.0))
    c1 = tl.arange(0, BLOCK_C1)
    c1m = c1 < CHI1
    u = tl.load(u_ptr + c1, mask=c1m, other=0.0)
    v = tl.load(v_ptr + c1, mask=c1m, other=0.0)
    ker_r = ca_r * u + cs_r * v
    ker_i = ca_i * u + cs_i * v
    fcw = tl.load(fcw_ptr + c1[:, None] * CHI1 + c1[None, :],
                  mask=c1m[:, None] & c1m[None, :], other=0.0)
    fcb = tl.load(fcb_ptr + c1, mask=c1m, other=0.0)
    fr = tl.sum(fcw * ker_r[None, :], axis=1) + fcb
    fi = tl.sum(fcw * ker_i[None, :], axis=1) + fcb
    msg_kernel = tl.math.atan2(fi, fr) * m

    # ---- accumulate dx[i] = [msg_kernel, x1] ----
    tl.atomic_add(dx_ptr + i * (CHI1 + H) + c1, msg_kernel, mask=c1m)
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    x1 = tl.load(x1_ptr + e * H + h, mask=hm, other=0.0)
    tl.atomic_add(dx_ptr + i * (CHI1 + H) + CHI1 + h, x1 * m, mask=hm)

    # ---- vector message ----
    xh2 = (tl.load(xh_ptr + j * 3 * H + H + h, mask=hm, other=0.0)
           * tl.load(rbfh_ptr + e * 3 * H + H + h, mask=hm, other=0.0)) * inv_sqrt_3
    xh3 = (tl.load(xh_ptr + j * 3 * H + 2 * H + h, mask=hm, other=0.0)
           * tl.load(rbfh_ptr + e * 3 * H + 2 * H + h, mask=hm, other=0.0))
    coef = inv_sqrt_h * m
    for d in range(0, 3):
        vj = tl.load(vec_ptr + j * 3 * H + d * H + h, mask=hm, other=0.0)
        rd = tl.load(r_ptr + e * 3 + d)
        vmsg = (vj * xh2 + xh3 * rd) * coef
        tl.atomic_add(dvec_ptr + i * 3 * H + d * H + h, vmsg, mask=hm)


@triton.jit
def _bwd_kernel(xh_ptr, vec_ptr, rbfh_ptr, x1_ptr, qr_ptr, qi_ptr, a_ptr,
                r_ptr, mask_ptr, idx_i_ptr, idx_j_ptr,
                Kr_ptr, Ki_ptr, u_ptr, v_ptr, fcw_ptr, fcb_ptr,
                gdx_ptr, gdvec_ptr,
                gxh_ptr, gvec_ptr, grbfh_ptr, gx1_ptr, gqr_ptr, gqi_ptr,
                ga_ptr, gr_ptr,
                H, HCHI, CHI1, CHI2, C, inv_sqrt_3, inv_sqrt_h,
                HAS_MASK: tl.constexpr,
                BLOCK_H: tl.constexpr, BLOCK_HCHI: tl.constexpr,
                BLOCK_C1: tl.constexpr, BLOCK_C2: tl.constexpr):
    e = tl.program_id(0)
    i = tl.load(idx_i_ptr + e)
    j = tl.load(idx_j_ptr + e)
    if HAS_MASK:
        m = tl.load(mask_ptr + e)
    else:
        m = 1.0

    # ---- recompute forward chi chain ----
    c2 = tl.arange(0, BLOCK_C2)
    c2m = c2 < CHI2
    cc = tl.arange(0, BLOCK_HCHI)
    ccm = cc < HCHI
    qr = tl.load(qr_ptr + e * HCHI + cc, mask=ccm, other=0.0)
    qi = tl.load(qi_ptr + e * HCHI + cc, mask=ccm, other=0.0)
    kr_bias = tl.zeros([BLOCK_C2], dtype=tl.float32)
    ki_bias = tl.zeros([BLOCK_C2], dtype=tl.float32)
    for cidx in range(0, C):
        kr_bias += tl.load(Kr_ptr + cidx * CHI2 + c2, mask=c2m, other=0.0)
        ki_bias += tl.load(Ki_ptr + cidx * CHI2 + c2, mask=c2m, other=0.0)
    Krm = tl.load(Kr_ptr + (C + cc[:, None]) * CHI2 + c2[None, :],
                  mask=ccm[:, None] & c2m[None, :], other=0.0)
    Kim = tl.load(Ki_ptr + (C + cc[:, None]) * CHI2 + c2[None, :],
                  mask=ccm[:, None] & c2m[None, :], other=0.0)
    conv_r = tl.sum(qr[:, None] * Krm - qi[:, None] * Kim, axis=0) + kr_bias
    conv_i = tl.sum(qr[:, None] * Kim + qi[:, None] * Krm, axis=0) + ki_bias
    a = tl.load(a_ptr + e * CHI2 + c2, mask=c2m, other=0.0)
    ca_r = tl.sum(conv_r * a)
    ca_i = tl.sum(conv_i * a)
    cs_r = tl.sum(tl.where(c2m, conv_r, 0.0))
    cs_i = tl.sum(tl.where(c2m, conv_i, 0.0))
    c1 = tl.arange(0, BLOCK_C1)
    c1m = c1 < CHI1
    u = tl.load(u_ptr + c1, mask=c1m, other=0.0)
    v = tl.load(v_ptr + c1, mask=c1m, other=0.0)
    ker_r = ca_r * u + cs_r * v
    ker_i = ca_i * u + cs_i * v
    fcw = tl.load(fcw_ptr + c1[:, None] * CHI1 + c1[None, :],
                  mask=c1m[:, None] & c1m[None, :], other=0.0)
    fcb = tl.load(fcb_ptr + c1, mask=c1m, other=0.0)
    fr = tl.sum(fcw * ker_r[None, :], axis=1) + fcb
    fi = tl.sum(fcw * ker_i[None, :], axis=1) + fcb

    # ---- backward through atan2 -> fc_mps -> rank-1 -> conv -> q ----
    gk = tl.load(gdx_ptr + i * (CHI1 + H) + c1, mask=c1m, other=0.0) * m
    denom = fr * fr + fi * fi + 1e-30
    gfr = -gk * fi / denom
    gfi = gk * fr / denom
    # fc_mps^T
    gker_r = tl.sum(fcw * gfr[:, None], axis=0)
    gker_i = tl.sum(fcw * gfi[:, None], axis=0)
    gca_r = tl.sum(gker_r * u)
    gcs_r = tl.sum(tl.where(c1m, gker_r * v, 0.0))
    gca_i = tl.sum(gker_i * u)
    gcs_i = tl.sum(tl.where(c1m, gker_i * v, 0.0))
    gconv_r = gca_r * a + gcs_r
    gconv_i = gca_i * a + gcs_i
    gconv_r = tl.where(c2m, gconv_r, 0.0)
    gconv_i = tl.where(c2m, gconv_i, 0.0)
    ga = conv_r * gca_r + conv_i * gca_i
    tl.store(ga_ptr + e * CHI2 + c2, tl.where(c2m, ga, 0.0), mask=c2m)
    gqr = tl.sum(Krm * gconv_r[None, :] + Kim * gconv_i[None, :], axis=1)
    gqi = tl.sum(Krm * gconv_i[None, :] - Kim * gconv_r[None, :], axis=1)
    tl.store(gqr_ptr + e * HCHI + cc, gqr, mask=ccm)
    tl.store(gqi_ptr + e * HCHI + cc, gqi, mask=ccm)

    # ---- x1 part of agg ----
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    gx1 = tl.load(gdx_ptr + i * (CHI1 + H) + CHI1 + h, mask=hm, other=0.0) * m
    tl.store(gx1_ptr + e * H + h, gx1, mask=hm)

    # ---- vector message backward ----
    rbf2 = tl.load(rbfh_ptr + e * 3 * H + H + h, mask=hm, other=0.0)
    rbf3 = tl.load(rbfh_ptr + e * 3 * H + 2 * H + h, mask=hm, other=0.0)
    xhj2 = tl.load(xh_ptr + j * 3 * H + H + h, mask=hm, other=0.0)
    xhj3 = tl.load(xh_ptr + j * 3 * H + 2 * H + h, mask=hm, other=0.0)
    xh2 = xhj2 * rbf2 * inv_sqrt_3
    xh3 = xhj3 * rbf3
    coef = inv_sqrt_h * m
    gxh2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    gxh3 = tl.zeros([BLOCK_H], dtype=tl.float32)
    for d in range(0, 3):
        gv = tl.load(gdvec_ptr + i * 3 * H + d * H + h, mask=hm, other=0.0) * coef
        vj = tl.load(vec_ptr + j * 3 * H + d * H + h, mask=hm, other=0.0)
        rd = tl.load(r_ptr + e * 3 + d)
        # d/dvec_j and d/dr accumulate; d/dxh2 += gv*vj; d/dxh3 += gv*rd
        tl.atomic_add(gvec_ptr + j * 3 * H + d * H + h, gv * xh2, mask=hm)
        gxh2 += gv * vj
        gxh3 += gv * rd
        gr_d = tl.sum(tl.where(hm, gv * xh3, 0.0))
        tl.atomic_add(gr_ptr + e * 3 + d, gr_d)
    # chain to xh (blocks 1,2) and rbfh (blocks 1,2); atomic over j for xh
    tl.atomic_add(gxh_ptr + j * 3 * H + H + h, gxh2 * rbf2 * inv_sqrt_3, mask=hm)
    tl.atomic_add(gxh_ptr + j * 3 * H + 2 * H + h, gxh3 * rbf3, mask=hm)
    tl.store(grbfh_ptr + e * 3 * H + H + h, gxh2 * xhj2 * inv_sqrt_3, mask=hm)
    tl.store(grbfh_ptr + e * 3 * H + 2 * H + h, gxh3 * xhj3, mask=hm)


def _pow2(x):
    return triton.next_power_of_2(max(int(x), 2))


class _FusedEquiMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xh, vec, rbfh, x1, qr, qi, a_dia, r_ij, edge_mask,
                idx_i, idx_j, Kr, Ki, u, v, fc_w, fc_b, chi1, chi2, head):
        N, H3 = xh.shape
        H = H3 // 3
        E = rbfh.shape[0]
        HCHI = qr.shape[1]
        C = HCHI // head
        dx = torch.zeros(N, chi1 + H, dtype=xh.dtype, device=xh.device)
        dvec = torch.zeros(N, 3, H, dtype=xh.dtype, device=xh.device)
        args = dict(H=H, HCHI=HCHI, CHI1=chi1, CHI2=chi2, C=C,
                    inv_sqrt_3=1.0 / math.sqrt(3.0),
                    inv_sqrt_h=1.0 / math.sqrt(H),
                    HAS_MASK=edge_mask is not None,
                    BLOCK_H=_pow2(H), BLOCK_HCHI=_pow2(HCHI),
                    BLOCK_C1=_pow2(chi1), BLOCK_C2=_pow2(chi2))
        mask_arg = edge_mask if edge_mask is not None else rbfh  # dummy ptr
        if E > 0:
            _fwd_kernel[(E,)](xh, vec, rbfh, x1, qr, qi, a_dia, r_ij,
                              mask_arg, idx_i, idx_j, Kr, Ki, u, v, fc_w, fc_b,
                              dx, dvec, **args)
        ctx.save_for_backward(xh, vec, rbfh, x1, qr, qi, a_dia, r_ij,
                              edge_mask if edge_mask is not None else rbfh.new_empty(0),
                              idx_i, idx_j, Kr, Ki, u, v, fc_w, fc_b)
        ctx.meta = (H, HCHI, chi1, chi2, C, edge_mask is not None)
        return dx, dvec

    @staticmethod
    def backward(ctx, gdx, gdvec):
        (xh, vec, rbfh, x1, qr, qi, a_dia, r_ij, mask, idx_i, idx_j,
         Kr, Ki, u, v, fc_w, fc_b) = ctx.saved_tensors
        H, HCHI, chi1, chi2, C, has_mask = ctx.meta
        E = rbfh.shape[0]
        gxh = torch.zeros_like(xh)
        gvec = torch.zeros_like(vec)
        grbfh = torch.zeros_like(rbfh)  # blocks 1,2 written; block 0 stays 0
        gx1 = torch.empty_like(x1)
        gqr = torch.empty_like(qr)
        gqi = torch.empty_like(qi)
        ga = torch.empty_like(a_dia)
        gr = torch.zeros_like(r_ij)
        if E > 0:
            args = dict(H=H, HCHI=HCHI, CHI1=chi1, CHI2=chi2, C=C,
                        inv_sqrt_3=1.0 / math.sqrt(3.0),
                        inv_sqrt_h=1.0 / math.sqrt(H),
                        HAS_MASK=has_mask,
                        BLOCK_H=_pow2(H), BLOCK_HCHI=_pow2(HCHI),
                        BLOCK_C1=_pow2(chi1), BLOCK_C2=_pow2(chi2))
            mask_arg = mask if has_mask else rbfh
            _bwd_kernel[(E,)](xh, vec, rbfh, x1, qr, qi, a_dia, r_ij,
                              mask_arg, idx_i, idx_j, Kr, Ki, u, v, fc_w, fc_b,
                              gdx.contiguous(), gdvec.contiguous(),
                              gxh, gvec, grbfh, gx1, gqr, gqi, ga, gr, **args)
        else:
            gx1.zero_(); gqr.zero_(); gqi.zero_(); ga.zero_()
        return (gxh, gvec, grbfh, gx1, gqr, gqi, ga, gr,
                None, None, None, None, None, None, None, None, None,
                None, None, None)


def fused_equi_message(layer, xh, vec, rbfh, edge_vector, edge_index,
                       edge_mask):
    """Fused replacement for EquiMessagePassing.propagate (inference, fp32).

    ``layer`` is the EquiMessagePassing module; the scale GEMM and the
    diagonal MLP run in cuBLAS/torch here (autograd handles their weights'
    chain), the rest of the message + aggregation is one Triton kernel.
    Returns (dx, dvec) with dx already [N, chi1+H] as the reference
    scatter produces.
    """
    assert layer.reduce_mode == "sum", "fused message requires sum reduction"
    j, i = edge_index[0], edge_index[1]
    H = layer.hidden_channels
    x1 = xh.index_select(0, j)[:, :H] * rbfh[:, :H]
    qr, qi = torch.split(layer.scale(x1), layer.hidden_channels_chi, dim=-1)
    a_dia = layer.activation(layer.diagonal(rbfh))
    norm_factor = math.sqrt(H // layer.head)
    Kr = (layer.kernel_real / norm_factor).reshape(-1, layer.chi2)
    Ki = (layer.kernel_imag / norm_factor).reshape(-1, layer.chi2)
    u = torch.matmul(layer.dia.weight, layer.diachi1)
    v = layer.dia.weight.sum(1) + layer.dia.bias
    mask_flat = edge_mask.reshape(-1).contiguous() if edge_mask is not None else None
    return _FusedEquiMessage.apply(
        xh.contiguous(), vec.contiguous(),
        rbfh.contiguous(), x1.contiguous(), qr.contiguous(), qi.contiguous(),
        a_dia.contiguous(), edge_vector.contiguous(), mask_flat,
        i.contiguous(), j.contiguous(),
        Kr.contiguous(), Ki.contiguous(), u.contiguous(), v.contiguous(),
        layer.fc_mps.weight.contiguous(), layer.fc_mps.bias.contiguous(),
        layer.chi1, layer.chi2, layer.head)
