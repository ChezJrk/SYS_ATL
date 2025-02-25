from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from exo import proc, instr, Procedure, DRAM, compile_procs_to_strings
from exo.libs.memories import MDRAM, MemGenError, StaticMemory, DRAM_STACK
from exo.libs.externs import *
from exo.stdlib.scheduling import *

mock_registers = 0


class MOCK(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        assert len(shape) == 1 and int(shape[0]) == 16
        global mock_registers
        if mock_registers > 0:
            raise MemGenError("Cannot allocate more than one mock register")
        mock_registers += 1

        return f"static {prim_type} {new_name}[16];"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        global mock_registers
        mock_registers -= 1
        return ""


# Testing to make sure free is inserted correcctly
def test_free(compiler):
    @proc
    def foo():
        x: f32[16] @ MOCK
        for i in seq(0, 16):
            x[i] = 3.0
        y: f32[16] @ MOCK
        for i in seq(0, 16):
            y[i] = 0.0

    compiler.compile(foo)


def test_free2(compiler):
    @proc
    def foo():
        x: f32[16] @ MOCK
        for i in seq(0, 16):
            y: R[16] @ MOCK
            y[i] = 2.0

    compiler.compile(foo)


def test_free3(compiler):
    @proc
    def foo():
        x: f32[16] @ MOCK
        for i in seq(0, 16):
            y: R[16] @ MOCK
            x[i] = 2.0

    with pytest.raises(MemGenError, match="Cannot allocate"):
        compiler.compile(foo)


old_split = repeat(divide_loop)


# Tests for constness


def test_const_local_buffer(golden, compiler):
    @proc
    def callee(N: size, A: [f32][N]):
        for i in seq(0, N):
            A[i] = 0.0

    @proc
    def caller():
        A: f32[10]
        callee(10, A)

    cc, hh = compile_procs_to_strings([caller], "test.h")
    assert f"{hh}{cc}" == golden

    compiler.compile(caller)


def test_const_local_window(golden, compiler):
    @proc
    def callee(N: size, A: [f32][N]):
        for i in seq(0, N):
            A[i] = 0.0

    @proc
    def caller():
        A: f32[100]
        callee(10, A[10:20])

    cc, hh = compile_procs_to_strings([caller], "test.h")
    assert f"{hh}{cc}" == golden

    compiler.compile(caller)


# --- Start Blur Test ---


def gen_blur() -> Procedure:
    @proc
    def blur(
        n: size,
        m: size,
        k_size: size,
        image: R[n, m],
        kernel: R[k_size, k_size],
        res: R[n, m],
    ):
        for i in seq(0, n):
            for j in seq(0, m):
                res[i, j] = 0.0
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, k_size):
                    for l in seq(0, k_size):
                        if (
                            i + k >= 1
                            and i + k - n < 1
                            and j + l >= 1
                            and j + l - m < 1
                        ):
                            res[i, j] += kernel[k, l] * image[i + k - 1, j + l - 1]

    return blur


def _test_blur(compiler, tmp_path, blur):
    fn = compiler.compile(blur)

    k_size = 5

    image = np.asarray(Image.open(Path(__file__).parent / "input.png"), dtype="float32")

    x = np.linspace(-1, 1, k_size + 1)
    kern1d = np.diff(np.random.normal(x))
    kern2d = np.outer(kern1d, kern1d)
    kern = np.asarray(kern2d / kern2d.sum(), dtype=np.float32)

    res = np.zeros_like(image)

    fn(None, *image.shape, k_size, image, kern, res)

    out = Image.fromarray(res.astype(np.uint8))
    out.save(tmp_path / "out.png")


def test_simple_blur(compiler, tmp_path):
    blur = gen_blur()
    _test_blur(compiler, tmp_path, blur)


def test_simple_blur_split(compiler, tmp_path):
    @proc
    def simple_blur_split(
        n: size,
        m: size,
        k_size: size,
        image: R[n, m],
        kernel: R[k_size, k_size],
        res: R[n, m],
    ):
        for i in seq(0, n):
            for j1 in seq(0, m / 2):
                for j2 in seq(0, 2):
                    res[i, j1 * 2 + j2] = 0.0
        for i in seq(0, n):
            for j1 in seq(0, m / 2):
                for j2 in seq(0, 2):
                    for k in seq(0, k_size):
                        for l in seq(0, k_size):
                            if (
                                i + k >= 1
                                and i + k - n < 1
                                and j1 * 2 + j2 + l >= 1
                                and j1 * 2 + j2 + l - m < 1
                            ):
                                res[i, j1 * 2 + j2] += (
                                    kernel[k, l] * image[i + k - 1, j1 * 2 + j2 + l - 1]
                                )

    _test_blur(compiler, tmp_path, simple_blur_split)


def test_split_blur(compiler, tmp_path):
    blur = gen_blur()

    blur = old_split(blur, "j", 4, ["j1", "j2"])
    blur = old_split(blur, "i#1", 4, ["i1", "i2"])

    _test_blur(compiler, tmp_path, blur)


def test_reorder_blur(compiler, tmp_path):
    blur = gen_blur()

    blur = reorder_loops(blur, "k l")
    blur = reorder_loops(blur, "i j")

    _test_blur(compiler, tmp_path, blur)


def test_unroll_blur(compiler, tmp_path):
    blur = gen_blur()

    blur = old_split(blur, "j", 4, ["j1", "j2"])
    blur = repeat(unroll_loop)(blur, "j2")

    _test_blur(compiler, tmp_path, blur)


# --- End Blur Test ---


# --- conv1d test ---
def test_conv1d(compiler):
    @proc
    def conv1d(n: size, m: size, r: size, x: R[n], w: R[m], res: R[r]):
        for i in seq(0, r):
            res[i] = 0.0
        for i in seq(0, r):
            for j in seq(0, n):
                if j < i + 1 and j >= i - m + 1:
                    res[i] += x[j] * w[i - j]

    n_size = 5
    m_size = 3
    r_size = n_size + m_size - 1
    x = np.array([0.2, 0.5, -0.4, 1.0, 0.0], dtype=np.float32)
    w = np.array([0.6, 1.9, -2.2], dtype=np.float32)
    res = np.zeros(shape=r_size, dtype=np.float32)

    fn = compiler.compile(conv1d)
    fn(None, n_size, m_size, r_size, x, w, res)

    np.testing.assert_almost_equal(
        res, np.array([0.12, 0.68, 0.27, -1.26, 2.78, -2.2, 0], dtype=np.float32)
    )


# ------- Nested alloc test for normal DRAM ------


def test_alloc_nest(compiler, tmp_path):
    @proc
    def alloc_nest(
        n: size, m: size, x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM
    ):
        for i in seq(0, n):
            rloc: R[m] @ DRAM
            xloc: R[m] @ DRAM
            yloc: R[m] @ DRAM
            for j in seq(0, m):
                xloc[j] = x[i, j]
            for j in seq(0, m):
                yloc[j] = y[i, j]
            for j in seq(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in seq(0, m):
                res[i, j] = rloc[j]

    # Write effect printing to a file
    (tmp_path / f"{alloc_nest.name()}_effect.atl").write_text(str(alloc_nest))

    x = np.array([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]], dtype=np.float32)
    y = np.array([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]], dtype=np.float32)
    res = np.zeros_like(x)

    fn = compiler.compile(alloc_nest)
    fn(None, *x.shape, x, y, res)

    np.testing.assert_almost_equal(
        res, np.array([[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], dtype=np.float32)
    )


# ------- Nested alloc test for custom malloc DRAM ------


def test_alloc_nest_malloc(compiler):
    @proc
    def alloc_nest_malloc(
        n: size, m: size, x: R[n, m] @ MDRAM, y: R[n, m] @ MDRAM, res: R[n, m] @ MDRAM
    ):
        for i in seq(0, n):
            rloc: R[m] @ MDRAM
            xloc: R[m] @ MDRAM
            yloc: R[m] @ MDRAM
            for j in seq(0, m):
                xloc[j] = x[i, j]
            for j in seq(0, m):
                yloc[j] = y[i, j]
            for j in seq(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in seq(0, m):
                res[i, j] = rloc[j]

    x = np.array([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]], dtype=np.float32)
    y = np.array([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]], dtype=np.float32)
    res = np.zeros_like(x)

    root_dir = Path(__file__).parent.parent
    lib = compiler.compile(
        alloc_nest_malloc,
        include_dir=str(root_dir / "src/exo/libs"),
        additional_file=str(root_dir / "src/exo/libs/custom_malloc.c"),
    )

    # Initialize custom malloc here
    lib.init_mem()
    lib(None, *x.shape, x, y, res)

    np.testing.assert_almost_equal(
        res, np.array([[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], dtype=np.float32)
    )


def test_unary_neg(compiler):
    @proc
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    res = np.zeros_like(x)

    fn = compiler.compile(negate_array)
    fn(None, res.shape[0], x, res)

    np.testing.assert_almost_equal(res, -x)


def test_bool1(compiler):
    @proc
    def foo(x: bool):
        pass

    @proc
    def bool1(a: bool, b: bool):
        assert b == True

        foo(False)
        x: f32
        if b == True:
            x = 0.0
        else:
            x = 1.0

        if a == b:
            x = 0.0

        if False:
            x = 0.0

        x: f32
        if a:
            x = 0.0

    compiler.compile(bool1, compile_only=True)


def test_sin1(compiler):
    @proc
    def sin1(x: f32):
        x = sin(x)

    buf = np.array([4.0], dtype=np.float32)
    out = np.sin(buf)

    fn = compiler.compile(sin1)
    fn(None, buf)

    np.testing.assert_almost_equal(buf, out)


def test_relu1(compiler):
    @proc
    def relu1(x: f32):
        x = relu(x)

    actual = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    expected = actual * (actual > 0)

    fn = compiler.compile(relu1)
    fn(None, actual)

    np.testing.assert_almost_equal(actual, expected)


def test_select1(compiler):
    @proc
    def select1(x: f32):
        zero: f32
        zero = 0.0
        two: f32
        two = 2.0
        # x < zero ? 2.0 : x
        x = select(x, zero, two, x)

    actual = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    expected = (actual < 0) * 2.0 + (actual >= 0) * actual

    fn = compiler.compile(select1)
    fn(None, actual)

    np.testing.assert_almost_equal(actual, expected)


##
# Tests for const-correctness


def test_const_buffer_parameters(golden, compiler):
    @proc
    def memcpy(N: size, A: f32[N], B: f32[N]):
        for i in seq(0, N):
            A[i] = B[i]

    memcpy_b = rename(set_window(memcpy, "B", True), "memcpy_b")
    memcpy_ab = rename(set_window(memcpy_b, "A", True), "memcpy_ab")

    c_file, h_file = compile_procs_to_strings([memcpy, memcpy_b, memcpy_ab], "test.h")
    code = f"{h_file}\n{c_file}"

    assert code == golden


# Tests for static memory


def test_static_memory_check(compiler):
    @proc
    def callee():
        pass

    @proc
    def caller():
        x: R
        if 1 < 2:
            y: R @ StaticMemory
        for i in seq(0, 8):
            callee()

    with pytest.raises(
        MemGenError, match="Cannot generate static memory in non-leaf procs"
    ):
        compiler.compile(caller)


# Tests for NO exo_floor_div


def test_no_exo_floor_div_after_divide_loop_with_guard(golden):
    @proc
    def foo(N: size, x: f32[N]):
        for i in seq(0, N):
            x[i] = 0.0

    foo = divide_loop(foo, foo.find_loop("i"), 8, ("io", "ii"))

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    code0 = f"{h_file}\n{c_file}"

    foo = cut_loop(foo, foo.find_loop("io"), "((N + 7) / (8)) - 1")

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    code1 = f"{h_file}\n{c_file}"

    foo = divide_loop(foo, foo.find_loop("io"), 4, ("ioo", "ioi"))

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    code2 = f"{h_file}\n{c_file}"

    code = f"{code0}\n{code1}\n{code2}\n"

    assert code == golden


def test_no_exo_floor_div_triangular_access(golden):
    @proc
    def foo(N: size, x: f32[N, N]):
        for ii in seq(0, N % 4):
            for joo in seq(0, (ii + N / 4 * 4) / 16):
                x[ii, joo] = 0.0

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    code = f"{h_file}\n{c_file}"

    assert code == golden


# Tests for CIR


def test_CIR_USub(golden):
    @proc
    def foo(N: size, x: f32[N]):
        for i in seq(0, N):
            x[-i + N - 1] = 0.0

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    code = f"{h_file}\n{c_file}"

    assert code == golden


def test_pragma_parallel_loop(golden):
    @proc
    def foo(x: i8[10]):
        for i in par(0, 10):
            y: i8[10] @ DRAM_STACK
            x[i] = y[i]

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_const_type_coercion(compiler):
    # 16777216 = 1 << 24 = integer precision limit for f32
    @proc
    def foo(x: f64[1], y: f32[1]):
        x[0] = (16777216.0 + 1.0) + y[0]

    # all consts should coerce to f32
    # so, 16777216 + 1 should yield 16777216 again
    x = np.array([3.0], dtype=np.float64)
    y = np.array([1.0], dtype=np.float32)
    fn = compiler.compile(foo)
    fn(None, x, y)
    assert x[0] == 16777216


def test_coercion_to_i8(golden):
    @proc
    def foo():
        a: i8
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_ui8(golden):
    @proc
    def foo():
        a: ui8
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_ui16(golden):
    @proc
    def foo():
        a: ui16
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_i32(golden):
    @proc
    def foo():
        a: i32
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_f16(golden):
    @proc
    def foo():
        a: f16
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_f32(golden):
    @proc
    def foo():
        a: f32
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_f64(golden):
    @proc
    def foo():
        a: f64
        a = a + 3

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_coercion_to_index(golden):
    @proc
    def foo():
        for x in seq(0, 6):
            pass

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_target_another_exo_library(compiler, tmp_path, golden):
    @proc
    def foo(n: size, x: f32[n]):
        for i in seq(0, n):
            x[i] = 1.0

    foo_sched = divide_loop(foo, foo.body()[0], 4, ("io", "ii"), tail="cut")
    foo_compile = foo_sched.compile_c(tmp_path, "foo")

    @proc
    def bar(n: size, y: f32[n]):
        for i in seq(0, n):
            y[i] = 1.0

    def using_make_instr():
        foo_instr = make_instr(foo, "foo(NULL, {n}, {x_data});", '#include "foo.h"')
        return replace(bar, bar.find_loop("i"), foo_instr)

    def using_instr_dec():
        @instr("foo(NULL, {n}, {x_data});", '#include "foo.h"')
        def foo(n: size, x: f32[n]):
            for i in seq(0, n):
                x[i] = 1.0

        return replace(bar, bar.find_loop("i"), foo)

    for func in using_make_instr, using_instr_dec:
        optimized_bar = func()

        foo_c, foo_h = compile_procs_to_strings([foo_sched], "foo.h")
        bar_c, bar_h = compile_procs_to_strings([optimized_bar], "bar.h")

        assert f"{foo_h}\n{foo_c}\n{bar_h}\n{bar_c}" == golden

        compiler.compile(optimized_bar, additional_file="foo.c")


def test_memcpy_instr(compiler, golden):
    @instr("memcpy({dst}, {src}, {n} * sizeof(float));", "#include <string.h>")
    def memcpy(n: size, dst: f32[n], src: f32[n]):
        for i in seq(0, n):
            dst[i] = src[i]

    @proc
    def bar(n: size, dst: f32[n], src: f32[n]):
        for i in seq(0, n):
            dst[i] = src[i]

    optimized_bar = replace(bar, bar.body()[0], memcpy)

    bar_c, bar_h = compile_procs_to_strings([optimized_bar], "bar.h")
    assert f"{bar_c}\n{bar_h}" == golden

    fn = compiler.compile(optimized_bar)

    n_size = 5
    src = np.array([float(i) for i in range(n_size)], dtype=np.float32)
    dst = np.zeros(shape=n_size, dtype=np.float32)

    fn(None, n_size, dst, src)

    expected = np.array([float(i) for i in range(n_size)], dtype=np.float32)

    np.testing.assert_almost_equal(dst, expected)
    np.testing.assert_almost_equal(src, expected)


def test_window_of_window_codegen(compiler):
    @proc
    def bar(n: size, dst: f32[n, n, n]):
        assert n >= 8
        w1 = dst[2:, 3, 1:]
        w2 = w1[0:, 4]  # dst[2:, 3, 5]
        for n in seq(0, 4):
            # Set dst[2:6, 3, 5] to 42
            w2[n] = 42.0
        w3 = w1[1, 4:]  # dst[3, 3, 5:]
        for n in seq(0, 2):
            # Set dst[3, 3, 5:7] to 137
            w3[n] = 137.0

    fn = compiler.compile(bar)

    for n_size in (8, 10):
        dst = np.zeros(shape=(n_size, n_size, n_size), dtype=np.float32)
        fn(None, n_size, dst)
        expected = np.zeros(shape=(n_size, n_size, n_size), dtype=np.float32)
        for n in range(0, 4):
            expected[2 + n, 3, 5] = 42.0
        for n in range(0, 2):
            expected[3, 3, 5 + n] = 137.0
        np.testing.assert_almost_equal(dst, expected)
