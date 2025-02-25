from __future__ import annotations

import pytest

import numpy as np

from exo import proc, config, instr
from exo.libs.memories import GEMM_SCRATCH
from exo.stdlib.scheduling import SchedulingError

# ------- Interpreter tests ---------


def test_mat_mul(compiler):
    @proc
    def rank_k_reduce(
        K: size,
        A: f32[6, K],
        B: f32[K, 16],
        C: f32[6, 16],
    ):
        for i in seq(0, 6):
            for j in seq(0, 16):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    fn = compiler.compile(rank_k_reduce)

    K = 8
    A = np.arange(6 * K, dtype=np.float32).reshape((6, K))
    B = np.arange(K * 16, dtype=np.float32).reshape((K, 16))
    C1 = np.zeros(6 * 16, dtype=np.float32).reshape((6, 16))
    C2 = np.zeros(6 * 16, dtype=np.float32).reshape((6, 16))

    fn(None, K, A, B, C1)
    rank_k_reduce.interpret(K=K, A=A, B=B, C=C2)
    assert (C1 == C2).all()


def test_reduce_add(compiler):
    @proc
    def acc(N: size, A: f32[N], acc: f32):
        acc = 0
        for i in seq(0, N):
            acc += A[i]

    fn = compiler.compile(acc)

    n = 3
    A = np.arange(n, dtype=np.float32)
    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, n, A, x)
    acc.interpret(N=n, A=A, acc=y)
    assert x == y


def test_scope1(compiler):
    @proc
    def foo(res: f32):
        a: f32
        a = 1
        for i in seq(0, 4):
            a: f32
            a = 2
        res = a

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, x)
    foo.interpret(res=y)
    assert x == y


def test_scope2(compiler):
    @proc
    def foo(res: f32):
        a: f32
        a = 1
        for i in seq(0, 4):
            a = 2
        res = a

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, x)
    foo.interpret(res=y)
    assert x == y


def test_empty_seq(compiler):
    @proc
    def foo(res: f32):
        for i in seq(0, 0):
            res = 1

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, x)
    foo.interpret(res=y)
    assert x == y


def test_cond(compiler):
    @proc
    def foo(res: f32, p: bool):
        if p:
            res = 1
        else:
            res = 2

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, x, False)
    foo.interpret(res=y, p=False)
    assert x == y


def test_call(compiler):
    @proc
    def bar(res: f32):
        res = 3

    @proc
    def foo(res: f32):
        res = 2
        bar(res)
        res += 1

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, x)
    foo.interpret(res=y)
    assert x == y


def test_window_assert(compiler):
    @proc
    def foo(
        n: size,
        m: size,
        src: [f32][n, m],
        dst: [f32][n, 16],
    ):
        assert n <= 16
        assert m <= 16

        for i in seq(0, n):
            for j in seq(0, m):
                dst[i, j] = src[i, j]

    n = 6
    m = 8
    src = np.arange(n * m, dtype=np.float32).reshape((n, m))
    dst = np.zeros(n * 16, dtype=np.float32).reshape((n, 16))

    foo.interpret(n=n, m=m, src=src, dst=dst)
    assert (dst[:, :8] == src).all()


def test_window_stmt1(compiler):
    @proc
    def foo(n: size, A: f32[n, 16], C: f32[n]):
        B = A[:, 0]
        for i in seq(0, n):
            C[i] = B[i]

    fn = compiler.compile(foo)

    n = 6
    A = np.arange(n * 16, dtype=np.float32).reshape((n, 16))
    C1 = np.arange(n, dtype=np.float32)
    C2 = np.arange(n, dtype=np.float32)

    fn(None, n, A, C1)
    foo.interpret(n=n, A=A, C=C2)

    assert (C1 == C2).all()


def test_window_stmt2(compiler):
    @proc
    def foo(n: size, A: f32[n], B: f32[n], C: f32[2 * n]):
        for i in seq(0, n):
            C[i] = A[i]
        for i in seq(n, 2 * n):
            C[i] = B[i - n]

    fn = compiler.compile(foo)

    n = 6
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    C1 = np.zeros(2 * n, dtype=np.float32)
    C2 = np.zeros(2 * n, dtype=np.float32)

    fn(None, n, A, B, C1)
    foo.interpret(n=n, A=A, B=B, C=C2)
    assert (C1 == C2).all()


def test_window_stmt3(compiler):
    @proc
    def foo(A: f32[8], res: f32):
        B = A[4:]
        res = B[0]

    fn = compiler.compile(foo)

    A = np.arange(8, dtype=np.float32)
    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn(None, A, x)
    foo.interpret(A=A, res=y)
    assert x[0] == 4 and x == y


# TODO: discuss
# error can be better here
def test_window_stmt4(compiler):
    @proc
    def foo(A: f32[8], C: [f32][4]):
        B = A[4:]
        C = B[:]


def test_stride_simple1(compiler):
    @proc
    def bar(s0: stride, s1: stride, B: [i8][3, 4]):
        assert stride(B, 0) == s0
        assert stride(B, 1) == s1
        pass

    @proc
    def foo(A: i8[3, 4]):
        bar(stride(A, 0), stride(A, 1), A[:, :])

    fn = compiler.compile(foo)

    A = np.arange(3 * 4, dtype=np.int8).reshape((3, 4))

    fn(None, A)
    foo.interpret(A=A)


def test_stride_simple2(compiler):
    @proc
    def bar(s0: stride, s1: stride, B: [i8][1, 1]):
        assert stride(B, 0) == s0
        assert stride(B, 1) == s1
        pass

    @proc
    def foo(A: [i8][3, 4]):
        bar(stride(A, 0), stride(A, 1), A[0:1, 1:2])

    fn = compiler.compile(foo)

    A = np.arange(6 * 8, dtype=np.int8).reshape((6, 8))

    fn(None, A[::2, ::2])
    foo.interpret(A=A[::2, ::2])


def test_stride1(compiler):
    @proc
    def foo(A: [i8][3, 2, 3]):
        assert stride(A, 0) == 20
        assert stride(A, 1) == 5 * 2
        assert stride(A, 2) == 1 * 2
        pass

    fn = compiler.compile(foo)

    A = np.arange(3 * 4 * 5, dtype=np.int8).reshape((3, 4, 5))

    fn(None, A[::1, ::2, ::2])
    foo.interpret(A=A[::1, ::2, ::2])


def test_stride2(compiler):
    @proc
    def foo(A: [i8][2, 4, 2]):
        assert stride(A, 0) == 20 * 2
        assert stride(A, 1) == 5 * 1
        assert stride(A, 2) == 1 * 3
        pass

    fn = compiler.compile(foo)

    A = np.arange(3 * 4 * 5, dtype=np.int8).reshape((3, 4, 5))

    fn(None, A[::2, ::1, ::3])
    foo.interpret(A=A[::2, ::1, ::3])


# TODO: discuss
# updating param within stride conditional triggers validation error
def test_branch_stride1(compiler):
    @proc
    def bar(B: [i8][3, 4], res: f32):
        if stride(B, 0) == 8:
            res = 1

    @proc
    def foo(A: i8[3, 4], res: f32):
        bar(A[:, :], res)


# but this is okay:
def test_branch_stride2(compiler):
    @proc
    def bar(B: [i8][3, 4], res: f32):
        if stride(B, 0) == 8:
            res = 1

    @proc
    def foo(A: i8[3, 4], res: f32):
        bar(A, res)


# so is this
def test_branch_stride3(compiler):
    @proc
    def bar(B: [i8][3, 4], res: f32):
        a: f32
        a = 0
        if stride(B, 0) == 8:
            a = 1
        res = a

    @proc
    def foo(A: i8[3, 4], res: f32):
        bar(A[:, :], res)


def test_bounds_err_interp():
    with pytest.raises(TypeError):

        @proc
        def foo(N: size, A: f32[N], res: f32):
            a: f32
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)


def test_precond_interp_simple():
    with pytest.raises(AssertionError):

        @proc
        def foo(N: size, A: f32[N], res: f32):
            assert N == 4
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)


def test_precond_interp_stride():
    with pytest.raises(AssertionError):

        @proc
        def foo(A: f32[1, 8]):
            assert stride(A, 0) == 8
            pass

        A = np.arange(16, dtype=np.float32).reshape((1, 16))
        foo.interpret(A=A[:, ::2])


def new_config():
    @config
    class Config:
        a: f32
        b: f32

    return Config


def test_config(compiler):
    Config = new_config()

    @proc
    def foo(x: f32):
        Config.a = 32.0
        x = Config.a

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    foo.interpret(x=x)
    assert x == 32.0


def test_config_nested(compiler):
    Config = new_config()

    @proc
    def bar(x: f32):
        x = Config.a + Config.b

    @proc
    def foo(x: f32):
        Config.a = 32.0
        Config.b = 16.0
        bar(x)

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    foo.interpret(x=x)
    assert x == 48.0


def test_par_bad():
    with pytest.raises(TypeError):

        @proc
        def foo(x: f32[10], acc: f32):
            for i in par(0, 10):
                acc += x[i]

        x = np.arange(10, dtype=np.float32)
        a = np.zeros(1, dtype=np.float32)

        foo.interpret(x=x, acc=a)


def test_par_good():
    @proc
    def foo(x: f32[10]):
        for i in par(0, 10):
            x[i] = 1

    x = np.zeros(10, dtype=np.float32)

    foo.interpret(x=x)
    assert (x == np.ones(10, dtype=np.float32)).all()


def test_built_in():
    @instr("")
    def four_wide_vector_add(m: size, A: [f64][m], B: [f64][m], C: [f64][m]):
        assert m >= 4
        for i in seq(0, 4):
            C[i] = A[i] + B[i]

    @proc
    def dumb_vector_add(n: size, A: f64[n], B: f64[n], C: f64[n]):
        assert n >= 5
        four_wide_vector_add(n - 1, A[1:], B[1:], C[1:])

    @proc
    def slightly_smarter_vector_add(n: size, A: f64[n], B: f64[n], C: f64[n]):
        assert (n % 4) == 0
        assert n >= 8
        for j in seq(0, n / 4):
            four_wide_vector_add(
                4,
                A[j * 4 : (j * 4) + 4],
                B[j * 4 : (j * 4) + 4],
                C[j * 4 : (j * 4) + 4],
            )

    A = np.array([1] * 5, dtype=np.float64)
    B = np.array([2] * 5, dtype=np.float64)
    C = np.zeros(5, dtype=np.float64)

    dumb_vector_add.interpret(n=5, A=A, B=B, C=C)
    assert (C == np.array([0, 3, 3, 3, 3], dtype=np.float64)).all()

    A = np.array([1] * 8, dtype=np.float64)
    B = np.array([2] * 8, dtype=np.float64)
    C = np.zeros(8, dtype=np.float64)
    slightly_smarter_vector_add.interpret(n=8, A=A, B=B, C=C)
    assert (C == np.array([3] * 8, dtype=np.float64)).all()
