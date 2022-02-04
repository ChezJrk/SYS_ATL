from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.x86 import *


# This is the reference code we _actually_ want to schedule.
@proc
def SSYRK_SIMPLE(M: size, K: size, A: f32[M, K], C: f32[M, M]):
    assert M >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):  # row i
        for j in seq(0, M):  # column j
            if j >= i:
                for k in seq(0, K):
                    C[i, j] += A[i, k] * A[j, k]


# row-major, upper-triangular, C := C + A @ A^T.
# noinspection PyPep8Naming
@proc
def SSYRK(M: size, K: size, A: f32[M, K], C: f32[M, M]):
    assert M >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):  # row i
        for j in seq(0, M):  # column j
            if j >= i:
                c_acc: R[16]
                for k in seq(0, 16):
                    c_acc[k] = 0.0
                for ko in seq(0, K / 16):
                    for ki in seq(0, 16):
                        c_acc[ki] += A[i, 16 * ko + ki] * A[j, 16 * ko + ki]
                if K % 16 > 0:
                    for ki in seq(0, K % 16):
                        c_acc[ki] += (A[i, 16 * (K / 16) + ki] *
                                      A[j, 16 * (K / 16) + ki])
                for k in seq(0, 16):
                    C[i, j] += c_acc[k]


SSYRK.unsafe_assert_eq(SSYRK_SIMPLE)

SSYRK = SSYRK.rename('systl_ssyrk')
SSYRK_WINDOW = (
    SSYRK.rename('systl_ssyrk_window')
        .set_window('A', True)
        .set_window('C', True)
)

ssyrk_edge_kernel = (
    SSYRK_WINDOW
        .rename('ssyrk_edge_kernel')
        .lift_alloc('c_acc: _', n_lifts=3)
        .expand_dim('c_acc: _', 'M', 'j')
        .expand_dim('c_acc: _', 'M', 'i')
        .set_memory('c_acc', AVX512)

        .fission_after('for k in _: _ #0', n_lifts=3)
        .fission_after('if K % _ > 0: _', n_lifts=3)
        .replace_all(mm512_set0_ps)
        .replace_all(mm512_reduce_add_ps)

        .stage_expr('A_vec', 'A[_] #0', memory=AVX512)
        .stage_expr('At_vec', 'A[_] #1', memory=AVX512)
        .replace_all(mm512_loadu_ps)
        .replace_all(mm512_fmadd_ps)
        .simplify()

        .bound_and_guard('for ki in _: _')
        .stage_expr('A_vec_mask', 'A[_] #2', memory=AVX512, n_lifts=2)
        .stage_expr('At_vec_mask', 'A[_] #3', memory=AVX512, n_lifts=2)
        .replace_all(mm512_maskz_loadu_ps)
        .replace_all(mm512_mask_fmadd_ps)
        .simplify()

        .fission_after('for ko in _: _', n_lifts=3)
        .lift_if('if K % _ > 0: _', n_lifts=3)

        .partial_eval(6)

        # TODO: need a way of hoisting a for past an if. This is a hack!
        .unroll('i')
        .unroll('j')
        .simplify()

        # TODO: slow!!
        .repeat(Procedure.fuse_loop, 'for ko in _: _ #0', 'for ko in _: _ #1')
)

if __name__ == '__main__':
    print(ssyrk_edge_kernel)

__all__ = ['SSYRK', 'ssyrk_edge_kernel']
