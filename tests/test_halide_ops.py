from __future__ import annotations


from exo import proc
from exo.platforms.x86 import *

from exo.stdlib.scheduling import *
from exo.stdlib.halide_scheduling_ops import *


@proc
def blur1d_compute_root(n: size, consumer: i8[n], inp: i8[n + 6]):
    producer: i8[n + 1]
    for i in seq(0, n + 1):
        producer[i] = (
            inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]
        ) / 6.0

    for i in seq(0, n):
        consumer[i] = (producer[i] + producer[i + 1]) / 2.0


def test_schedule_blur1d(golden):
    p = blur1d_compute_root
    procs = []

    loop = p.find_loop("i #1")
    p = fuse_at(p, "producer", "consumer", loop)
    p = rename(p, "blur1d_compute_at_store_root")
    procs.append(p)

    loop = p.find_loop("i")
    p_bounds = (0, "i", 0, 2)
    p = store_at(p, "producer", loop)
    p = rename(p, "blur1d_compute_at")
    procs.append(p)

    p = unroll_loop(p, "ii")
    for i in range(2):
        p = inline_assign(p, p.find("consumer[_] = _").prev())
    p = delete_buffer(p, "producer: _")
    p = simplify(p)
    p = rename(p, "blur1d_inline")
    procs.append(p)

    print("\n\n".join([str(p) for p in procs]))
    assert "\n\n".join([str(p) for p in procs]) == golden


@proc
def blur2d_compute_root(n: size, consumer: i8[n, n], sin: i8[n + 1, n + 1]):
    assert n % 4 == 0
    producer: i8[n + 1, n + 1]
    for i in seq(0, n + 1):
        for j in seq(0, n + 1):
            producer[i, j] = sin[
                i, j
            ]  # just a placeholder since sine can't evalute on index exprs

    for i in seq(0, n):
        for j in seq(0, n):
            consumer[i, j] = (
                producer[i, j]
                + producer[i, j + 1]
                + producer[i + 1, j]
                + producer[i + 1, j + 1]
            ) / 4.0


def test_schedule_blur2d(golden):
    p = blur2d_compute_root
    procs = []

    p = fuse_at(p, "producer", "consumer", p.find_loop("i #1"))
    p = rename(p, "blur2d_compute_at_i_store_root")
    procs.append(p)

    p = blur2d_compute_root
    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = rename(p, "blur2d_compute_at_j_store_root")
    procs.append(p)

    p = blur2d_compute_root
    p = fuse_at(p, "producer", "consumer", p.find_loop("i #1"))
    p = store_at(p, "producer", p.find_loop("i"))
    p = rename(p, "blur2d_compute_at_i")
    procs.append(p)

    p = blur2d_compute_root
    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = store_at(p, "producer", p.find_loop("i"))
    p = simplify(p)
    p = rename(p, "blur2d_compute_at_j_store_at_i")
    procs.append(p)

    p = blur2d_compute_root
    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = store_at(p, "producer", p.find_loop("j"))
    p = unroll_loop(p, "ii")
    p = unroll_loop(p, "ji")
    for i in range(4):
        p = inline_assign(p, p.find("consumer[_] = _").prev())
    p = delete_buffer(p, "producer: _")
    p = rename(p, "blur2d_inline")
    procs.append(p)

    assert "\n\n".join([str(p) for p in procs]) == golden


def test_schedule_tiled_blur2d(golden):
    compute_root = blur2d_compute_root
    procs = []

    p_tiled = tile(
        compute_root,
        compute_root.find_loop("i #1"),
        compute_root.find_loop("j #1"),
        ["i", "ii"],
        ["j", "ji"],
        4,
        4,
        perfect=True,
    )

    p = p_tiled
    p = rename(p, "blur2d_tiled")
    procs.append(p)

    p = p_tiled
    p = fuse_at(p, "producer", "consumer", p.find_loop("i #1"))
    p = rename(p, "blur2d_tiled_compute_at_i")
    procs.append(p)

    p = p_tiled
    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = rename(p, "blur2d_tiled_compute_at_j")
    procs.append(p)

    p = p_tiled
    p = fuse_at(p, "producer", "consumer", p.find_loop("ii"))
    p = rename(p, "blur2d_tiled_compute_at_ii")
    procs.append(p)

    p = p_tiled
    p = fuse_at(p, "producer", "consumer", p.find_loop("ji"))
    p = rename(p, "blur2d_tiled_compute_at_ji")
    procs.append(p)

    p = p_tiled
    p = fuse_at(p, "producer", "consumer", p.find_loop("ji"))
    p = store_at(p, "producer", p.find_loop("ji"))
    p = rename(p, "blur2d_tiled_compute_at_and_store_at_ji")
    procs.append(p)

    assert "\n\n".join([str(p) for p in procs]) == golden
