"""GreenLeaf: BF16 gate_up GEMM with SiLU*mul fused into the epilogue (Triton, SM120).

Opt-in with SGLANG_GL_FUSED_SWIGLU=1; used only when the token count is at least
SGLANG_GL_FUSED_SWIGLU_MIN_M (default 3072). Below that, cuBLAS plus the separate
act_and_mul kernel is faster on an RTX 5090.

The output matches F.linear followed by silu_and_mul bit for bit (measured with
0 differing elements at M from 2048 to 16384). The match holds because:
  * gate and up use separate fp32 accumulators, run mma.sync m16n8k16 over K in
    ascending order without split-K, and are rounded to bf16 like the GEMM output;
  * the epilogue runs the act_and_mul_kernel formula, compiled with
    --use_fast_math: g / (1 + __expf(-g)) * u in fp32 (ftz), then rounded to bf16.
"""

import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("SGLANG_GL_FUSED_SWIGLU", "0") == "1"
_MIN_M = int(os.environ.get("SGLANG_GL_FUSED_SWIGLU_MIN_M", "3072"))

# Config fixed from the K1 microbench (autotune picked it at every M). Tuning at
# serve time would re-tune on every new M.
_BM, _BN, _BK, _GM, _WARPS, _STAGES = 128, 64, 32, 8, 4, 4


@triton.jit
def _gate_up_silu_kernel(
    a_ptr, w_ptr, c_ptr, M, I, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(I, BN)
    gsz = GM * npn
    fm = (pid // gsz) * GM
    gm = min(npm - fm, GM)
    pm = fm + (pid % gsz) % gm
    pn = (pid % gsz) // gm
    rm = (pm * BM + tl.arange(0, BM)) % M
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a_ptr + rm[:, None] * K + rk[None, :]
    gp = w_ptr + rn[None, :] * K + rk[:, None]
    up = w_ptr + (rn[None, :] + I) * K + rk[:, None]
    ag = tl.zeros((BM, BN), dtype=tl.float32)
    au = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(ap)
        ag = tl.dot(a, tl.load(gp), ag)
        au = tl.dot(a, tl.load(up), au)
        ap += BK
        gp += BK
        up += BK
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    out = tl.inline_asm_elementwise(
        "{ .reg .f32 t, e, d, s;\n mul.ftz.f32 t, $1, 0fBFB8AA3B;\n ex2.approx.ftz.f32 e, t;\n"
        " add.ftz.f32 d, e, 0f3F800000;\n div.full.ftz.f32 s, $1, d;\n mul.ftz.f32 $0, s, $2; }",
        "=f,f,f", [g, u], dtype=tl.float32, is_pure=True, pack=1,
    )
    om = pm * BM + tl.arange(0, BM)
    tl.store(c_ptr + om[:, None] * I + rn[None, :], out.to(tl.bfloat16), mask=om[:, None] < M)


def can_use(x: torch.Tensor, gate_up_proj) -> bool:
    if not _ENABLED or x.ndim != 2 or x.shape[0] < _MIN_M:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    weight = gate_up_proj.weight
    return (
        gate_up_proj.bias is None
        and type(gate_up_proj.quant_method) is UnquantizedLinearMethod
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % _BK == 0
        and weight.shape[0] % (2 * _BN) == 0
    )


def gate_up_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """silu(x @ W_gate^T) * (x @ W_up^T), where weight = [W_gate; W_up] has shape [2I, K]."""
    M, K = x.shape
    I = weight.shape[0] // 2
    out = torch.empty((M, I), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, _BM) * triton.cdiv(I, _BN),)
    _gate_up_silu_kernel[grid](
        x, weight, out, M, I, K, BM=_BM, BN=_BN, BK=_BK, GM=_GM, num_warps=_WARPS, num_stages=_STAGES
    )
    return out
