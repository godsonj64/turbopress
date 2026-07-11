"""GPTQ / LDLQ error feedback for the rotated scalar quantizer.

Nearest rounding treats each weight independently, so a layer's output error
``(W - W_hat) x`` is a fixed bias with no attempt to cancel across input
channels. GPTQ (Frantar et al. 2022) / LDLQ (QuIP, Chee et al. 2023) instead
quantize the input channels *sequentially* and, after fixing each channel,
feed its rounding error forward into the not-yet-quantized channels weighted by
the layer's input Hessian ``H = E[x x^T]``. The remaining channels absorb the
error in the directions the data actually excites, so the end-to-end output
error ``tr((W - W_hat) H (W - W_hat)^T)`` is minimized rather than the raw
weight MSE.

This module runs LDLQ in the *rotated, equilibrated* coordinate system the rest
of turbopress uses:

    y = W x = (W D R^T)(R D^-1 x),   z := R D^-1 x

so the relevant Hessian is ``H_z = E[z z^T] = R D^-1 H D^-1 R^T``. The random
rotation R is exactly the incoherence preprocessing that makes LDLQ effective
(QuIP): it spreads the Hessian mass so no single channel dominates the
feedback. :func:`rotated_hessian` builds ``H_z`` from the raw activation
Hessian using the same ``RandomizedOrthogonal`` R and equilibration folds as
:class:`~turbopress.linear.QJLCorrectedLinear`.

The per-channel quantizer is the row-wise Lloyd-Max Gaussian codebook (one
scale per output row, fixed up front from the row RMS), so the stored form and
bit accounting are identical to the plain ``method="scalar"`` path -- LDLQ only
changes *which* codeword each weight rounds to.
"""

from __future__ import annotations

import torch
from torch import Tensor

from turbopress.codebooks import lloyd_max_gaussian
from turbopress.hadamard import RandomizedOrthogonal
from turbopress.quantizer import RowQuantized
from turbopress.trellis import TCQQuantized, Trellis, _viterbi, tcq_optimized_codebook

__all__ = ["rotated_hessian", "ldlq_quantize_rows", "ldlq_tcq_quantize_rows"]


def rotated_hessian(
    hessian: Tensor,
    rotation: RandomizedOrthogonal,
    inv_col_scale: Tensor | None = None,
) -> Tensor:
    """Map a raw input Hessian ``E[x x^T]`` into rotated/equilibrated space.

    Returns ``H_z = R D^-1 H D^-1 R^T`` where ``D^-1 = diag(inv_col_scale)``
    (identity when ``inv_col_scale`` is None) and R is ``rotation``. Uses the
    fast transform twice (``rotation`` applies R^T on the right of each row), so
    the cost is O(d^2 log d) rather than forming R densely.
    """
    d = hessian.shape[-1]
    if hessian.shape != (d, d):
        raise ValueError(f"hessian must be square [d, d], got {tuple(hessian.shape)}")
    h = hessian.to(torch.float32)
    if inv_col_scale is not None:
        inv = inv_col_scale.to(h.device, torch.float32)
        h = h * inv[:, None] * inv[None, :]
    # rotation(M) == M R^T (applied along the last dim). Apply on both sides:
    #   A = H' R^T ;  H_z = R H' R^T = (A^T R^T) = rotation(A.T).
    a = rotation(h)
    hz = rotation(a.transpose(-1, -2).contiguous())
    return 0.5 * (hz + hz.transpose(-1, -2))  # symmetrize away fp drift


def _nearest_codes(z: Tensor, codebook: Tensor) -> Tensor:
    thresholds = (codebook[:-1] + codebook[1:]) / 2.0
    return torch.bucketize(z.contiguous(), thresholds)


def _hinv_cholesky(
    hessian_z: Tensor, percdamp: float, dev: torch.device
) -> tuple[Tensor, Tensor]:
    """Upper Cholesky factor U of the damped ``H^-1`` (GPTQ feedback matrix).

    Returns ``(U, dead)`` where ``dead`` marks non-positive diagonal channels
    (the caller zeroes the corresponding weight columns, GPTQ-style). The
    per-column update ``err_j = (w_j - q_j) / U_jj``, ``w_k -= err_j U_jk``
    (k > j) implements exact LDLQ; the block form
    ``E = (W_B - Q_B) U_BB^-1``, ``W_{>B} -= E U_{B,>B}`` is its block-LDL
    generalization (identical to the scalar sweep when the block size is 1).
    """
    d = hessian_z.shape[-1]
    h = hessian_z.to(torch.float32).clone().to(dev)
    diag = torch.arange(d, device=dev)
    diagvals = h[diag, diag]
    dead = diagvals <= 0
    if dead.any():
        h[diag[dead], diag[dead]] = 1.0
    damp = percdamp * diagvals[~dead].mean() if (~dead).any() else percdamp
    h[diag, diag] += damp
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    u = torch.linalg.cholesky(0.5 * (h_inv + h_inv.T), upper=True)
    return u, dead


def ldlq_quantize_rows(
    w_rot: Tensor,
    hessian_z: Tensor,
    bits: int,
    percdamp: float = 0.01,
    block_size: int = 128,
) -> RowQuantized:
    """LDLQ-quantize each row of ``w_rot`` against Hessian ``hessian_z``.

    ``w_rot`` is [n_rows, d] (already rotated/equilibrated); ``hessian_z`` is
    the [d, d] input Hessian in the *same* coordinates (see
    :func:`rotated_hessian`). Per-row scales are the row RMS, fixed before the
    sweep. Returns a :class:`RowQuantized` identical in form to the plain
    scalar quantizer, so downstream storage accounting is unchanged.
    """
    if w_rot.ndim != 2:
        raise ValueError(f"expected [n_rows, dim], got {tuple(w_rot.shape)}")
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n_rows, d = w_rot.shape
    if hessian_z.shape != (d, d):
        raise ValueError(
            f"hessian_z must be [{d}, {d}], got {tuple(hessian_z.shape)}"
        )
    dev = w_rot.device
    codebook = lloyd_max_gaussian(bits)[0].to(torch.float32).to(dev)

    w = w_rot.to(torch.float32).clone()
    row_scale = w.pow(2).mean(dim=1).sqrt()
    nonzero = row_scale > 0
    safe = torch.where(nonzero, row_scale, torch.ones_like(row_scale))[:, None]

    # Prepare the inverse-Hessian Cholesky (upper-triangular), GPTQ-style.
    h_inv, dead = _hinv_cholesky(hessian_z, percdamp, dev)
    if dead.any():
        w[:, dead] = 0.0

    codes = torch.zeros(n_rows, d, dtype=torch.uint8, device=dev)
    q_full = torch.zeros_like(w)

    for i1 in range(0, d, block_size):
        i2 = min(i1 + block_size, d)
        w_block = w[:, i1:i2].clone()
        q_block = torch.zeros_like(w_block)
        err_block = torch.zeros_like(w_block)
        hinv_block = h_inv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = w_block[:, j]
            d_jj = hinv_block[j, j]
            idx = _nearest_codes(col / safe[:, 0], codebook)
            q_col = safe[:, 0] * codebook[idx]
            q_block[:, j] = q_col
            codes[:, i1 + j] = idx.to(torch.uint8)
            err = (col - q_col) / d_jj
            w_block[:, j:] -= err[:, None] * hinv_block[j, j:][None, :]
            err_block[:, j] = err
        q_full[:, i1:i2] = q_block
        w[:, i2:] -= err_block @ h_inv[i1:i2, i2:]

    scales = torch.where(nonzero, safe[:, 0], torch.zeros_like(safe[:, 0]))
    codes[~nonzero] = 0
    return RowQuantized(codes=codes, scales=scales, codebook=codebook, bits=bits)


def ldlq_tcq_quantize_rows(
    w_rot: Tensor,
    hessian_z: Tensor,
    bits: int,
    n_states: int = 64,
    percdamp: float = 0.01,
    block_size: int = 128,
    codebook: Tensor | None = None,
    codebook_mode: str = "optimized",
    scale_iters: int = 0,
) -> TCQQuantized:
    """Block-LDLQ error feedback with the trellis as the block quantizer.

    The QTIP-style combination of the pipeline's two strongest methods:
    columns are processed in blocks of ``block_size``; each block is
    trellis-coded *jointly* (the Viterbi encoder continues from the end state
    of the previous block, so the concatenated path bits decode exactly like
    a plain TCQ stream -- storage format and rate are unchanged), and each
    block's quantization error is fed forward into the not-yet-quantized
    columns through the block-LDL factors of the damped ``H_z^-1``, exactly
    as GPTQ/LDLQ does per column. With a diagonal Hessian and a single block
    this reduces to plain ``tcq_quantize_rows(scale_iters=0)`` (proven in
    tests). Per-row scales are fixed *before* each sweep so the feedback
    stays consistent with the final reconstruction; ``scale_iters`` re-runs
    the whole sweep after a least-squares refit of the scales against the
    original rows (the refinement ``tcq_quantize_rows`` applies). It
    defaults to 0 here: measured on SmolLM2-135M at 2 bits, the refit
    interacts destructively with the feedback (the refit fits scales to
    codes that encode feedback-*compensated* rows; the mis-fit scale clips
    the small codebook and the feedback amplifies the error -- KL 2.24 ->
    4.19). Pass ``codebook`` to override the trellis-optimized default
    (e.g. a codebook rounded to its stored fp16 precision).
    """
    if w_rot.ndim != 2:
        raise ValueError(f"expected [n_rows, dim], got {tuple(w_rot.shape)}")
    if not 1 <= bits <= 7:
        raise ValueError(f"tcq bits must be in [1, 7], got {bits}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    n_rows, d = w_rot.shape
    if hessian_z.shape != (d, d):
        raise ValueError(f"hessian_z must be [{d}, {d}], got {tuple(hessian_z.shape)}")
    dev = w_rot.device
    trellis = Trellis(n_states)
    if codebook is None:
        if codebook_mode == "optimized":
            codebook = tcq_optimized_codebook(bits, n_states)
        elif codebook_mode == "lloyd":
            codebook = lloyd_max_gaussian(bits + 1)[0].to(torch.float32)
        else:
            raise ValueError(
                f"codebook_mode must be 'optimized' or 'lloyd', got {codebook_mode!r}"
            )
    codebook = codebook.to(torch.float32).to(dev)

    w0 = w_rot.to(torch.float32).clone()
    u, dead = _hinv_cholesky(hessian_z, percdamp, dev)
    if dead.any():
        w0[:, dead] = 0.0

    row_scale = w0.pow(2).mean(dim=1).sqrt()
    nonzero = row_scale > 0
    safe = torch.where(nonzero, row_scale, torch.ones_like(row_scale))

    def sweep(safe: Tensor):
        w = w0.clone()
        level_codes = torch.empty(n_rows, d, dtype=torch.uint8, device=dev)
        path_bits = torch.empty(n_rows, d, dtype=torch.uint8, device=dev)
        member_codes = torch.empty(n_rows, d, dtype=torch.uint8, device=dev)
        state = torch.zeros(n_rows, dtype=torch.int64, device=dev)
        inv_safe = (1.0 / safe)[:, None]
        for i1 in range(0, d, block_size):
            i2 = min(i1 + block_size, d)
            lv, pb, mc, state = _viterbi(
                w[:, i1:i2] * inv_safe, codebook, trellis, start_state=state
            )
            level_codes[:, i1:i2] = lv
            path_bits[:, i1:i2] = pb
            member_codes[:, i1:i2] = mc
            if i2 < d:
                q_block = safe[:, None] * codebook[lv.long()]
                # Block-LDL feedback: E = (W_B - Q_B) U_BB^-1 ; W_{>B} -= E U_{B,>B}.
                err = torch.linalg.solve_triangular(
                    u[i1:i2, i1:i2], w[:, i1:i2] - q_block, upper=True, left=False
                )
                w[:, i2:] -= err @ u[i1:i2, i2:]
        return level_codes, path_bits, member_codes

    level_codes, path_bits, member_codes = sweep(safe)
    for _ in range(scale_iters):
        q = codebook[level_codes.long()]
        num = (w0 * q).sum(dim=1)
        den = (q * q).sum(dim=1)
        new_scale = torch.where(den > 0, num / den.clamp_min(1e-30), safe)
        safe = torch.where(new_scale > 0, new_scale, safe)
        level_codes, path_bits, member_codes = sweep(safe)

    scales = torch.where(nonzero, safe, torch.zeros_like(safe))
    return TCQQuantized(
        level_codes=level_codes,
        path_bits=path_bits,
        member_codes=member_codes,
        scales=scales,
        codebook=codebook,
        bits=bits,
        n_states=n_states,
    )
